from copy import deepcopy
import torch
import torch.nn.functional as F
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.quantization import quantize_, int4_weight_only, int8_weight_only
from torchao.prototype.awq.api import ObservedLinear, insert_awq_observer, awq_quant

def simple_test():
    class ToyLinearModel(torch.nn.Module):
        def __init__(self, m=512, n=256, k=128):
            super().__init__()
            self.linear1 = torch.nn.Linear(m, n, bias=False)
            self.linear2 = torch.nn.Linear(n, k, bias=False)
            self.linear3 = torch.nn.Linear(k, 1, bias=False)

        def example_inputs(self, batch_size, sequence_length=10, dtype=torch.bfloat16, device="cuda"):
            return [torch.randn(1, sequence_length, self.linear1.in_features, dtype=dtype, device=device) for j in range(batch_size)]

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            return x
        

    device = ("cuda")
    torch.manual_seed(34)
    dataset_size = 1000
    dtype = torch.bfloat16
    l1,l2,l3 = 512,256,128
    m = ToyLinearModel(l1,l2,l3).eval().to(dtype).to(device)
    m_bf16 = deepcopy(m)

    dataset = m.example_inputs(dataset_size,  dtype=dtype, device=device)
    calibration_data = dataset[:100]
    bf16_out = torch.cat([m_bf16(i.squeeze(0)) for i in dataset], dim=0)


    m_int4wo = deepcopy(m)
    quantize_(m_int4wo, int8_weight_only())
    int4wo_out = torch.cat([m_int4wo(i.squeeze(0)) for i in dataset])

    # calibrate
    quant_dtype = "int4"
    group_size = 128
    insert_awq_observer(m, quant_dtype, group_size, dtype, device)
    for example in calibration_data:
        m(example.to(device))
    # print('calibrated')

    # quantize
    is_observed_linear = lambda m, fqn: isinstance(m, ObservedLinear)
    scales = []
    quantize_(m, awq_quant(quant_dtype = quant_dtype, scale_list=scales, group_size = group_size), is_observed_linear)
    print(scales)
    awq_out = torch.cat([m(i.squeeze(0)) for i in dataset])
    # m = torch.compile(m, fullgraph=True)
    # compare accuracy
    awq_err = torch.sum(torch.abs(awq_out - bf16_out)).sum().item() / dataset_size
    int4wo_err = torch.sum(torch.abs(int4wo_out - bf16_out)).sum().item() / dataset_size
    print(f"AWQ error: {awq_err}")
    print(f"Int4WO error: {int4wo_err}")


from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import evaluate
from lm_eval.tasks import get_task_dict
from datasets import load_dataset
from tqdm import tqdm
import time

def create_batches_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def wikitext_eval(repo_id, quant, calibrate_size=100, eval_size=1000, device="cuda", precision=torch.bfloat16, max_length=2048, compile=False):
    print("Loading model ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=precision).to(device)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")
    # print(model)
    wikitext103 = load_dataset("wikitext", "wikitext-103-v1")
    wikitext103_train  = wikitext103["train"]

    if quant.startswith("awq"):
        quant_dtype = quant.split("-")[1]
        print(f"running {quant} calibration")
        t0 = time.time()
        quant_dtype = quant.split("-")[1]
        group_size = 128 if quant_dtype == "int4" else -1
        insert_awq_observer(model, quant_dtype, group_size, precision, device)
        wikitext103_calibration = wikitext103_train.select(range(calibrate_size))
        calibration_input_ids = [tokenizer.encode(text, return_tensors="pt").to(device=device) for text in wikitext103_calibration["text"]]

        for example in tqdm(wikitext103_calibration):
            inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs.input_ids.to(device=device)
            model(input_ids)

        print(f"time for calibration: {time.time() - t0:.02f} seconds")
        is_observed_linear = lambda m, fqn: isinstance(m, ObservedLinear)
        t0 = time.time()
        scales = []
        quantize_(model, awq_quant(quant_dtype=quant_dtype, group_size = group_size, scale_list=scales), is_observed_linear)
        print(f"time for quantization: {time.time() - t0:.02f} seconds")

        # print("scale distributions:")
        # for scale in scales:
        #     print(f"min: {scale.min().item():.02f}, max: {scale.max().item():.02f}, avg: {scale.mean().item():.02f}")
        # print(model)

    elif quant=="int8":
        print("running int8 quantization")
        quantize_(model, int8_weight_only())
        # print(model)

    elif quant=="int4":
        print("running int4 quantization")
        quantize_(model, int4_weight_only())
        # print(model)
    if compile:
        model = torch.compile(model)

    eval_data = wikitext103["train"].select(range(calibrate_size, min(calibrate_size+eval_size,len(wikitext103["train"]))))
    total_loss = 0.0
    total_tokens = 0
    print("Evaluating...")
    for example in tqdm(eval_data):
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to(device=device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        loss = 0 if torch.isnan(outputs.loss) else outputs.loss.item()
        total_loss += loss * input_ids.size(1)
        total_tokens += input_ids.size(1)

    ppl = torch.tensor(total_loss / total_tokens).exp().item()
    print(f"Perplexity: {ppl:.5f}")
    return ppl
    # int8 100,100: 5505.30371  
    # awq int8 100,100: 5546.76807
    # bf16 100,100: 5546.76807

awq = wikitext_eval("Xenova/llama2.c-stories15M","awq-int4", 100, 1000, compile=False)
int8 = wikitext_eval("Xenova/llama2.c-stories15M","awq-int8", 100, 1000, compile=False)
# print(f"wikitext perplexity on {10} sentences\nawq: {awq}\nint8wo: {int8}")
# simple_test()

