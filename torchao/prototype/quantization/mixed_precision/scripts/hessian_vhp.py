# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import transformers
from datasets import load_dataset
from torch.autograd.functional import vhp
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm


def group_product(xs, ys):
    return [torch.sum(x * y) for (x, y) in zip(xs, ys)]


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc.input_ids


# utilities to make nn.Module functional
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(mod, layer_id):
    orig_params = tuple(mod.parameters())
    # remove all the parameters in the model
    selected_params = []
    selected_params_names = []

    names = []
    for name, p in list(mod.named_parameters()):
        if name.startswith(
            "model.layers." + str(layer_id) + ".self_attn."
        ) or name.startswith("model.layers." + str(layer_id) + ".mlp."):
            selected_params.append(p)
            selected_params_names.append(name)
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names, selected_params, selected_params_names


def main(layer_id, checkpoint, max_seqlen, max_iter, nsamples):
    # use the functional model to load the weights back
    def load_weights(mod, names, params, selected_params, selected_params_names):
        for name, p in zip(names, params):
            if name.startswith(
                "model.layers." + str(layer_id) + ".self_attn."
            ) or name.startswith("model.layers." + str(layer_id) + ".mlp."):
                idx = selected_params_names.index(name)
                set_attr(mod, name.split("."), selected_params[idx])
            else:
                set_attr(mod, name.split("."), p)
        for name, param in mod.named_parameters():
            if param.requires_grad:
                print(f"Parameter {name} requires gradients.")

    # define the function to calculate the vhp
    def f(*new_params):
        load_weights(model, names, params, new_params, selected_params_names)
        model.zero_grad()
        outputs = model(inputs)
        logits = outputs.logits
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # to avoid aten::_scaled_dot_product_flash_attention_backward not implemented error
    with sdpa_kernel(SDPBackend.MATH):
        # have been tested models Llama-3-8B, Llama-2-7B, Mistral-7B, and stories110M
        model = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
        model = model.to(device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()

        # load calibration dataset
        trainloader, _ = get_wikitext2(128, 0, 2048, tokenizer)

        # make the model functional
        params, names, selected_params, selected_params_names = make_functional(
            model, layer_id
        )

        # make params regular Tensors instead of nn.Parameter
        params = tuple(p.detach() for p in params)

        # set requires_grad to True for the selected parameters
        selected_params_tuple = tuple(
            p.detach().requires_grad_() for p in selected_params
        )

        trace_history = []
        vhv_c_history = []

        for iteration in range(max_iter):
            print("iteration: ", iteration)

            # generate Rademacher random variables
            v = [torch.randint_like(p, high=2) for p in selected_params_tuple]
            for v_i in v:
                v_i[v_i == 0] = -1

            for i in tqdm(range(nsamples)):
                inputs, labels = trainloader[i]
                inputs = inputs.to(device)
                labels = labels.to(device)
                # if use testloader:
                # inputs = testloader[:, (i * max_seqlen) : ((i + 1) * max_seqlen)].to(device)
                # labels = testloader[:, (i * max_seqlen) : ((i + 1) * max_seqlen)].to(device)

                # get vector-Hessian product
                _, vH = vhp(f, selected_params_tuple, tuple(v))

                if i == 0:
                    TvH = [
                        torch.zeros(p.size()).to(device) for p in selected_params_tuple
                    ]
                TvH = [TvH1 + vH1 + 0.0 for TvH1, vH1 in zip(TvH, vH)]

            TvH = [TvH1 / float(nsamples) for TvH1 in TvH]
            # get vHv
            vHv = group_product(TvH, v)
            vHv_c = np.array([i.to(torch.float32).cpu().numpy() for i in vHv])
            vhv_c_history.append(vHv_c)
            trace = np.sum(np.abs(vHv_c))
            print("trace", trace)
            trace_history.append(trace)

        print("Iteration Done")
        print("Avg Hessian trace for layer", layer_id, "is:", np.mean(trace_history))
        print("trace_history,", trace_history)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate layer-wised Hessian trace leveraging torch's vhp function."
    )
    # TODO: make it a for loop for all the layer_ids to automatically calculate the Hessian trace for all the layers of a model
    parser.add_argument(
        "--layer_id",
        type=int,
        default=0,
        help="Which layer to compute the Hessian trace",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/tmp/Meta-Llama-3-8B",
        help="Path to load model",
    )
    parser.add_argument(
        "--max_seqlen", type=int, default=2048, help="Max sequence length"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="The number of iterations to calculate Hessian trace",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="The number of samples in calibration dataset",
    )
    args = parser.parse_args()
    main(args.layer_id, args.checkpoint, args.max_seqlen, args.max_iter, args.nsamples)
