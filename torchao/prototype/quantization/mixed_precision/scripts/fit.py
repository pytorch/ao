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
from tqdm import tqdm


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
    return trainloader, testenc


def cal_FIT(device, data, nsamples, model, max_iter, max_seqlen, criterion, num_layers):
    # store the history of trace for each layer
    estimated_history = []

    # store the history of mean trace for each layer
    estimated_mean = [[] for _ in range(num_layers)]
    trace = [0.0] * num_layers

    for iteration in range(max_iter):
        print("iteration: ", iteration)
        trace_tmp = [0.0] * num_layers

        for i in tqdm(range(nsamples)):
            inputs, targets = data[i]
            inputs = inputs.to(device)
            targets = targets.to(device)
            model.zero_grad()
            outputs = model(inputs)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            grads = torch.autograd.grad(loss, model.parameters())

            # Trace(Fisher Information Matrix) is calculated by the sum of the square of the gradient
            for layerid in range(num_layers):
                for (name, _), grad in zip(model.named_parameters(), grads):
                    if "." + str(layerid) + "." in name and (
                        "self_attn" in name or "mlp" in name
                    ):
                        trace_tmp[layerid] += torch.sum(grad * grad).item()

            # clean cache
            model.zero_grad()
            del grads
            torch.cuda.empty_cache()

        # calculate the mean of the trace on the calibration dataset
        for t in range(num_layers):
            trace[t] = trace_tmp[t] / float(nsamples)
            estimated_mean[t].append(trace[t])

        print("trace:", trace)
        estimated_history.append(trace)

    F_average = np.array([np.mean(i) for i in estimated_mean])
    return F_average, estimated_mean, estimated_history


def main(max_seqlen, checkpoint, nsamples, max_iter, num_layers):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # have been tested models Llama-3-8B, Llama-2-7B, Mistral-7B, and stories110M
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    model = model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    # load calibration dataset
    seed = 0
    trainloader, testloader = get_wikitext2(nsamples, seed, max_seqlen, tokenizer)

    F_average, estimated_mean, estimated_history = cal_FIT(
        device=device,
        data=trainloader,
        nsamples=nsamples,
        model=model,
        max_iter=max_iter,
        max_seqlen=max_seqlen,
        criterion=criterion,
        num_layers=num_layers,
    )
    print("Iteration Done")
    print("FIT scores for", num_layers, "layers:\n", F_average)
    print("estimated_mean:", estimated_mean)
    print("estimated_history:", estimated_history)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate layer-wised fish information matrix trace."
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
        help="The number of iterations to calculate FIT",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=32,
        help="The number of layers to calculate FIT.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="The number of samples in calibration dataset",
    )
    args = parser.parse_args()
    main(
        args.max_seqlen, args.checkpoint, args.nsamples, args.max_iter, args.num_layers
    )
