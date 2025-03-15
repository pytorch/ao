# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests

from torchao._models.llama.model import Transformer, prepare_inputs_for_model
from torchao._models.llama.tokenizer import get_tokenizer
from torchao.quantization.GPTQ_MT import Int4WeightOnlyGPTQQuantizer, MultiTensor
from torchao.quantization.utils import _lm_eval_available
from torchao.utils import is_fbcode

if is_fbcode():
    pytest.skip("Skipping the test in fbcode due to missing model and tokenizer files")

if _lm_eval_available:
    hqq_core = pytest.importorskip("hqq.core", reason="requires hqq")
    import lm_eval

    try:  # lm_eval version 0.4
        from lm_eval.evaluator import evaluate
        from lm_eval.models.huggingface import HFLM as eval_wrapper
        from lm_eval.tasks import get_task_dict
    except:  # lm_eval version 0.3
        from lm_eval import base, evaluator, tasks

        eval_wrapper = base.BaseLM
        get_task_dict = tasks.get_task_dict
        evaluate = evaluator.evaluate

    class InputRecorder(eval_wrapper):
        def __init__(
            self,
            tokenizer,
            calibration_seq_length,
            input_prep_func=None,
            pad_calibration_inputs=False,
            vocab_size=32000,
            pad_token=0,
            device="cpu",
        ):
            try:
                super().__init__()
            except TypeError:
                # lm_eval 0.4.2 removed the default init
                super().__init__("gpt2", device="cpu")

            self.tokenizer = tokenizer
            self._device = torch.device(device)
            self.vocab_size = vocab_size
            self._max_seq_length = calibration_seq_length
            self.calibration_seq_length = calibration_seq_length

            self.input_prep_func = (
                input_prep_func if input_prep_func is not None else lambda x: (x,)
            )

            self.pad_calibration_inputs = pad_calibration_inputs
            self.pad_token = pad_token

            self.inputs = []

        @property
        def eot_token_id(self):
            try:
                return self.tokenizer.eos_id()
            except:
                return self.tokenizer.eos_id

        @property
        def max_length(self):
            return self._max_seq_length

        @property
        def max_gen_toks(self):
            return 50

        @property
        def batch_size(self):
            return 1

        @property
        def device(self):
            return self._device

        def tok_encode(self, string: str, **kwargs):
            tokens = self.tokenizer.encode(string)
            if hasattr(self.tokenizer, "bos_id"):
                try:
                    tokens = [self.tokenizer.bos_id()] + tokens
                except:
                    tokens = [self.tokenizer.bos_id] + tokens
            return tokens

        def tok_decode(self, tokens):
            decoded = self.tokenizer.decode(tokens)
            return decoded

        def add_input(self, args):
            self.inputs.append(args)

        def record_inputs(
            self,
            calibration_tasks,
            calibration_limit,
        ):
            try:
                lm_eval.tasks.initialize_tasks()
            except:
                pass

            task_dict = get_task_dict(calibration_tasks)
            print("Obtaining GPTQ calibration inputs on: ", calibration_tasks)

            evaluate(
                self,
                task_dict,
                limit=calibration_limit,
            )
            return self

        def get_inputs(self):
            return self.inputs

        def _model_call(self, inps):
            inps = inps.squeeze(0)
            T = len(inps)
            if (
                # can't use inputs that are too short when padding disabled
                (T < self.calibration_seq_length and not self.pad_calibration_inputs)
                or
                # can't use inputs that actually use token we use for padding
                (self.pad_calibration_inputs and self.pad_token in inps)
            ):
                # give random output
                return torch.randn(
                    (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
                )

            # pad or truncate to the right size
            if T >= self.calibration_seq_length:
                inps = inps[: self.calibration_seq_length]
            else:
                inps = F.pad(inps, (self.pad_token, self.calibration_seq_length - T))

            inps = inps.unsqueeze(0)
            model_in = self.input_prep_func(inps)

            self.add_input(model_in)

            # output `something` with correct shape to keep eval going
            return torch.randn(
                (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
            )

        def _model_generate(self, context, max_length, eos_token_id):
            raise Exception("unimplemented")

    import logging
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class TransformerEvalWrapper(InputRecorder):
        """
        A wrapper class for GPTFast, providing integration with the lm-evaluation-harness library.
        """

        def __init__(
            self, model, tokenizer, max_seq_length, input_prep_func=None, device="cuda"
        ):
            super().__init__(tokenizer, None)
            self._model = model
            # self.tokenizer = tokenizer
            self._device = torch.device(device)
            self._max_seq_length = max_seq_length

            # need to take inps and convert to corrent input
            # for model
            self.input_prep_func = (
                input_prep_func if input_prep_func is not None else lambda x: (x,)
            )

        def _model_call(self, inps):
            # print("Entering _model_call")
            # print(f"Input shape: {inps.shape}")

            input = self.input_prep_func(inps)
            # print(f"Processed input shapes: {[x.shape for x in input]}")

            input = [x.to(self._device) for x in input]
            # print(f"Inputs moved to device: {self._device}")

            max_seq_length = min(max(inps.size()), self.max_length)
            # print(f"Max sequence length: {max_seq_length}")

            # print("Setting up caches")
            with torch.device(self._device):
                # print(f"Device: {self._device}")
                # print(f"Batch size: {self.batch_size}")
                # print(f"Max sequence length: {max_seq_length}")
                self._model.setup_caches(self.batch_size, max_seq_length)
            # print("Caches set up")

            # print("Running model")
            # torch.save(input, "input.pt")
            logits = self._model(*input)
            # print(f"Model run complete. Logits shape: {logits.shape}")
            return logits

        def _model_generate(self, context, max_length, eos_token_id):
            raise Exception("unimplemented")

        def run_eval(self, tasks, limit):
            logger.info(f"Starting evaluation on tasks: {tasks}")
            logger.info(f"Evaluation limit: {limit}")

            try:
                logger.info("Initializing lm_eval tasks")
                lm_eval.tasks.initialize_tasks()
            except Exception as e:
                logger.warning(f"Failed to initialize tasks: {e}")
                logger.info("Continuing without initialization")

            try:
                logger.info("Getting task dictionary")
                task_dict = get_task_dict(tasks)
                logger.info(f"Task dictionary: {task_dict}")
            except Exception as e:
                logger.error(f"Failed to get task dictionary: {e}")
                raise

            logger.info("Starting evaluation")
            start_time = time.time()

            try:
                with torch.no_grad():
                    result = evaluate(self, task_dict, limit=limit, verbosity="DEBUG")
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                raise

            end_time = time.time()
            logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")

            logger.info("Evaluation results:")
            for task, res in result["results"].items():
                print(f"{task}: {res}")

            return result


def test_gptq_mt():
    precision = torch.bfloat16
    device = "cuda"
    print("Loading model")
    checkpoint_path = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")
    model = Transformer.from_name(checkpoint_path.parent.name)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(dtype=precision, device="cpu")
    model.eval()
    print("Model loaded")
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), tokenizer_path
    tokenizer = get_tokenizer(  # pyre-ignore[28]
        tokenizer_path,
        "Llama-2-7b-chat-hf",
    )
    print("Tokenizer loaded")

    blocksize = 128
    percdamp = 0.01
    groupsize = 64
    calibration_tasks = ["wikitext"]
    calibration_limit = None
    calibration_seq_length = 100
    input_prep_func = prepare_inputs_for_model
    pad_calibration_inputs = False
    print("Recording inputs")
    inputs = (
        InputRecorder(
            tokenizer,
            calibration_seq_length,
            input_prep_func,
            pad_calibration_inputs,
            model.config.vocab_size,
            device="cpu",
        )
        .record_inputs(
            calibration_tasks,
            calibration_limit,
        )
        .get_inputs()
    )
    print("Inputs recorded")
    quantizer = Int4WeightOnlyGPTQQuantizer(
        blocksize,
        percdamp,
        groupsize,
    )

    model.setup_caches(max_batch_size=1, max_seq_length=calibration_seq_length)
    multi = [
        MultiTensor([inp for inp, _ in inputs]),
        MultiTensor([inds for _, inds in inputs]),
    ]
    print("Quantizing model")
    model = quantizer.quantize(model, multi).cuda()
    print("Model quantized")
    print("Saving model and fixing state dict")
    regular_state_dict = model.state_dict()  # defaultdict(torch.tensor)
    for key, value in model.state_dict().items():
        if isinstance(value, MultiTensor):
            regular_state_dict[key] = value.values[0]
        else:
            regular_state_dict[key] = value

    model = Transformer.from_name(checkpoint_path.parent.name)
    remove = [k for k in regular_state_dict if "kv_cache" in k]
    for k in remove:
        del regular_state_dict[k]

    model.load_state_dict(regular_state_dict, assign=True)
    torch.save(model.state_dict(), "model.pth")
    print("Running evaluation")
    TransformerEvalWrapper(
        model.to(device),  # quantized model needs to run on cuda
        tokenizer,
        model.config.block_size,
        prepare_inputs_for_model,
    ).run_eval(
        ["wikitext"],
        None,
    )


if __name__ == "__main__":
    run_tests()

# wikitext: {'word_perplexity,none': 12.523175352665858, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.6042723245990418, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.681919059499152, 'bits_per_byte_stderr,none': 'N/A', 'alias': 'wikitext'}
