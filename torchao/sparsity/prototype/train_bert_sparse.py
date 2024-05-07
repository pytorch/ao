import collections
import datasets
import evaluate
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor, SparseSemiStructuredTensorCUSPARSELT, SparseSemiStructuredTensorCUTLASS
from torch.ao.pruning import WeightNormSparsifier
import transformers
import torch.nn.functional as F

# force CUTLASS use if cuSPARSELt is not available
torch.manual_seed(100)
def preprocess_validation_function(examples, tokenizer):
    inputs = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def preprocess_train_function(examples, tokenizer):
    inputs = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs["offset_mapping"]
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, (offset, answer) in enumerate(zip(offset_mapping, answers)):
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def compute_metrics(start_logits, end_logits, features, examples):
    n_best = 20
    max_answer_length = 30
    metric = evaluate.load("squad")

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    # for example in tqdm(examples):
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0
                    # or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def measure_execution_time(model, batch_sizes, dataset):
    dataset_for_model = dataset.remove_columns(["example_id", "offset_mapping"])
    dataset_for_model.set_format("torch")
    batch_size_to_time_sec = {}
    for batch_size in batch_sizes:
        batch = {
            k: dataset_for_model[k][:batch_size].cuda()
            for k in dataset_for_model.column_names
        }

        with torch.no_grad():
            baseline_predictions = model(**batch)
            timer = benchmark.Timer(
                stmt="model(**batch)", globals={"model": model, "batch": batch}
            )
            p50 = timer.blocked_autorange().median * 1000
            batch_size_to_time_sec[batch_size] = p50

    return batch_size_to_time_sec
from torchao.sparsity.prototype.fast_sparse_training import sparsify24



class SemiSparseLinear(nn.Linear):

    def forward(self, x, sparsify_activations=False):
        w_sparse = sparsify24(self.weight, backend="cusparselt")
        return F.linear(x, w_sparse, self.bias)

    @classmethod
    def from_dense(cls, linear):
        mod = cls(linear.in_features, linear.out_features)
        mod.weight = linear.weight
        mod.bias = linear.bias
        return mod

def swap_linear_with_semi_sparse_linear_(model, config, current=""):
        name_to_child = dict(model.named_children())
        for name, child in name_to_child.items():
            if isinstance(child, torch.nn.Linear) and max(child.out_features, child.in_features) > 1024:
                setattr(model, name, SemiSparseLinear.from_dense(child))
                del child
            else:
                swap_linear_with_semi_sparse_linear_(child, config, current=f"{current}.{name}")

def swap_semi_sparse_linear_to_linear(model, config, current=""):
        name_to_child = dict(model.named_children())
        for name, child in name_to_child.items():
            if isinstance(child, SemiSparseLinear):
                new_module = nn.Linear(child.in_features, child.out_features)
                new_module.weight = nn.Parameter(to_sparse_semi_structured(child.weight))
                new_module.bias.data = child.bias.data
                setattr(model, name, new_module)
                del child
            else:
                swap_semi_sparse_linear_to_linear(child, config, current=f"{current}.{name}")

if __name__ == "__main__":
    # load model
    model_name = "bert-base-cased"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    print(f"Loading tokenizer: {model_name}")
    print(f"Loading model: {model_name}")

    # set up train and val dataset
    squad_dataset = datasets.load_dataset("squad")
    tokenized_squad_dataset = {}
    tokenized_squad_dataset["train"] = squad_dataset["train"].map(
        lambda x: preprocess_train_function(x, tokenizer), batched=True,
        # remove_columns=squad_dataset["train"].column_names,
    )
    tokenized_squad_dataset["validation"] = squad_dataset["validation"].map(
        lambda x: preprocess_validation_function(x, tokenizer),
        batched=True,
        remove_columns=squad_dataset["train"].column_names,
    )
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    swap_linear_with_semi_sparse_linear_(model, {})

    for name, mod in model.named_modules():
        if isinstance(mod, SemiSparseLinear):
            (packed, meta, packed_t, meta_t, bitmask) = torch._sparse_semi_structured_tile(
                mod.weight.data.cuda().to(torch.bfloat16), algorithm="", use_cutlass=True
            )
            out = SparseSemiStructuredTensorCUTLASS(
                mod.weight.data.shape,
                packed=packed,
                meta=meta,
                packed_t=packed_t,
                meta_t=meta_t,
                compressed_swizzled_bitmask=bitmask,
                requires_grad=False,
                fuse_transpose_cusparselt=True,
            )
            mod.weight = nn.Parameter(out)

    training_args = transformers.TrainingArguments(
        "trainer",
        num_train_epochs=1,
        lr_scheduler_type="constant",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        torch_compile=True,
        bf16=True,
        # optim="adamw_torch_fused",
        dataloader_drop_last=True, # since we compile, drop the last training batch because it does not have the right dims.
        dataloader_num_workers=8,
        logging_strategy="no",
        save_strategy="no"
    )

    trainer = transformers.Trainer(
        model,
        training_args,
        train_dataset=tokenized_squad_dataset["train"],
        eval_dataset=tokenized_squad_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # swap_semi_sparse_linear_to_linear(model, {})
    print("Evaluating")
    eval_args = transformers.TrainingArguments(
        "eval",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        bf16=True,
        dataloader_num_workers=8,
        logging_strategy="no",
        save_strategy="no"
    )

    trainer = transformers.Trainer(
        model,
        eval_args,
        train_dataset=tokenized_squad_dataset["train"],
        eval_dataset=tokenized_squad_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    predictions = trainer.predict(tokenized_squad_dataset["validation"])
    start_logits, end_logits = predictions.predictions
    metrics = compute_metrics(
        start_logits,
        end_logits,
        tokenized_squad_dataset["validation"],
        squad_dataset["validation"],
    )
    print(metrics)
