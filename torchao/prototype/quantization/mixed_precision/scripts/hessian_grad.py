import random

import numpy as np
import torch
import transformers
from datasets import load_dataset
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


def dataloader_hv_product(
    layerid, params, device, v, data, nsamples, model, max_seqlen, criterion
):
    model.zero_grad()
    THv = [torch.zeros(p.size()).to(device) for p in params]  # accumulate result

    # Freeze all the parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the parameters of attention and MLP layers in layer 0
    layer_ = model.model.layers[layerid]
    for param in layer_.self_attn.parameters():
        param.requires_grad = True
    for param in layer_.mlp.parameters():
        param.requires_grad = True

    for i in tqdm(range(nsamples)):
        torch.cuda.empty_cache()
        inputs, labels = data[i]
        inputs = inputs.to(device)
        labels = labels.to(device)
        # if use testloader:
        # inputs = data[:, (i * max_seqlen) : ((i + 1) * max_seqlen)].to(device)
        # labels = data[:, (i * max_seqlen) : ((i + 1) * max_seqlen)].to(device)
        model.zero_grad()
        outputs = model(inputs)
        logits = outputs.logits
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # get the first order gradients
        grads = torch.autograd.grad(loss, params, create_graph=True, only_inputs=True)

        # calculate Hessian vector product via Jac-vector product
        Hv = torch.autograd.grad(
            grads, params, grad_outputs=v, only_inputs=True, retain_graph=False
        )

        THv = [THv1 + Hv1 + 0.0 for THv1, Hv1 in zip(THv, Hv)]

        # clean cache
        model.zero_grad()
        del Hv
        del grads
        torch.cuda.empty_cache()

    THv = [THv1 / float(nsamples) for THv1 in THv]
    return THv


def cal_trace(
    layerid, params, device, data, nsamples, model, max_iter, max_seqlen, criterion
):
    vhv_c_history = []
    trace_history = []
    trace = 0.0

    for i in range(max_iter):
        print("iteration: ", i)

        # generate Rademacher random variables
        v = [torch.randint_like(p, high=2, device=device) for p in params]

        for v_i in v:
            v_i[v_i == 0] = -1

        # calculate Hessian vector product
        Hv = dataloader_hv_product(
            layerid, params, device, v, data, nsamples, model, max_seqlen, criterion
        )

        vHv = group_product(Hv, v)

        vHv_c = np.array([i.cpu().numpy() for i in vHv])

        vhv_c_history.append(vHv_c)

        trace = np.sum(vHv_c)

        trace_history.append(trace)
        print("trace,", trace)
        print("trace_history,", trace_history)
        print("vhv_c_history,", vhv_c_history)

    return np.mean(trace_history)


def main(layer_id, checkpoint, max_seqlen, max_iter, nsamples):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # to avoid aten::_scaled_dot_product_flash_attention_backward not implemented error
    with sdpa_kernel(SDPBackend.MATH):
        # have been tested models Llama-3-8B, Llama-2-7B, Mistral-7B, and stories110M
        model = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
        model = model.cuda()
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()

        # load calibration dataset
        seed = 0
        trainloader, testloader = get_wikitext2(128, seed, 2048, tokenizer)

        # calculate Hessian for only one layer each time
        params = []
        layer_ = model.model.layers[layer_id]
        for param in layer_.self_attn.parameters():
            params.append(param)
        for param in layer_.mlp.parameters():
            params.append(param)

        trace = cal_trace(
            layerid=layer_id,
            params=params,
            device=device,
            data=trainloader,
            nsamples=nsamples,
            model=model,
            max_iter=max_iter,
            max_seqlen=max_seqlen,
            criterion=criterion,
        )
        print("The trace of layer " + str(layer_id) + " is", trace)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate layer-wised Hessian trace leveraging autograd."
    )
    parser.add_argument(
        "--layer_id",
        type=int,
        default=0,
        help="Which layer to compute the trace and hessian",
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
