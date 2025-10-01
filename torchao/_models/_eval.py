# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import lm_eval
import torch
import torch.nn.functional as F

from torchao.quantization.GPTQ import MultiTensorInputRecorder

try:  # lm_eval version 0.4
    from lm_eval.evaluator import evaluate  # pyre-ignore[21]
    from lm_eval.models.huggingface import HFLM as eval_wrapper  # pyre-ignore[21]
    from lm_eval.tasks import get_task_dict  # pyre-ignore[21]
except:  # lm_eval version 0.3
    from lm_eval import base, evaluator, tasks

    eval_wrapper = base.BaseLM
    get_task_dict = tasks.get_task_dict
    evaluate = evaluator.evaluate


class TransformerEvalWrapper(eval_wrapper):
    """
    A wrapper class for GPTFast, providing integration with the lm-evaluation-harness library.
    """

    def __init__(
        self, model, tokenizer, max_seq_length=512, input_prep_func=None, device="cuda"
    ):
        try:
            super().__init__(device=device)
        except TypeError:
            # lm_eval 0.4.2 removed the default init
            super().__init__("gpt2", device="cpu")

        self._model = model
        self.tokenizer = tokenizer
        self._device = torch.device(device)
        self._max_seq_length = max_seq_length

        # need to take inps and convert to corrent input
        # for model
        self.input_prep_func = (
            input_prep_func if input_prep_func is not None else lambda x: (x,)
        )

    def _model_call(self, inps):
        # TODO: make batches work
        input = self.input_prep_func(inps)

        max_seq_length = min(max(inps.size()), self.max_length)
        with torch.device(self._device):
            if hasattr(self._model, "setup_caches"):
                self._model.setup_caches(self.batch_size, max_seq_length)
        output = self._model(*input)
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3CausalLMOutputWithPast,
        )

        if isinstance(output, (CausalLMOutputWithPast, Gemma3CausalLMOutputWithPast)):
            logits = output.logits
        else:
            logits = output
        return logits

    def run_eval(self, tasks, limit):
        try:
            lm_eval.tasks.initialize_tasks()
        except:
            pass

        task_dict = get_task_dict(tasks)
        print("Evaluating Model On: ", task_dict)
        with torch.no_grad():
            result = evaluate(
                self,
                task_dict,
                limit=limit,
            )
        for task, res in result["results"].items():
            print(f"{task}: {res}")
        return result

    @property
    def eot_token_id(self):
        try:
            return self.tokenizer.eos_id()
        except:
            try:
                return self.tokenizer.eos_id
            except:
                idx = self.tokenizer.all_special_tokens.index("<|endoftext|>")
                return self.tokenizer.all_special_ids[idx]

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

    def tok_decode(self, tokens, **kwargs):
        decoded = self.tokenizer.decode(tokens, **kwargs)
        return decoded

    def tok_encode(self, string: str, **kwargs):
        tokens = self.tokenizer.encode(string)
        if hasattr(self.tokenizer, "bos_id"):
            try:
                tokens = [self.tokenizer.bos_id()] + tokens
            except:
                tokens = [self.tokenizer.bos_id] + tokens
        return tokens


class LMEvalInputRecorder(TransformerEvalWrapper):
    def __init__(
        self,
        tokenizer,
        calibration_seq_length,
        input_prep_func=None,
        vocab_size=32000,
        pad_calibration_inputs=False,
        pad_token=0,
        device="cpu",
        base_input_recorder_class=MultiTensorInputRecorder,
    ):
        super().__init__(
            model=None,
            tokenizer=tokenizer,
            max_seq_length=calibration_seq_length,
            input_prep_func=input_prep_func,
            device=device,
        )
        self.vocab_size = vocab_size
        self.calibration_seq_length = calibration_seq_length

        self.pad_calibration_inputs = pad_calibration_inputs
        self.pad_token = pad_token

        # Initialize inputs as a list of two empty lists for input tensors and indices
        self.base_input_recorder = base_input_recorder_class()

    def record_inputs(self, calibration_tasks, calibration_limit):
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

    def get_inputs(self):  # for BC
        return self.get_recorded_inputs()

    def get_recorded_inputs(self):
        return self.base_input_recorder.get_recorded_inputs()

    def get_recorded_args_and_kwargs(self):
        return self.base_input_recorder.get_recorded_args_and_kwargs()

    def _model_call(self, inps):
        inps = inps.squeeze(0)
        T = len(inps)
        if (
            # Can't use inputs that are too short when padding is disabled
            (T < self.calibration_seq_length and not self.pad_calibration_inputs)
            or
            # Can't use inputs that actually use the token we use for padding
            (self.pad_calibration_inputs and self.pad_token in inps)
        ):
            # Give random output
            return torch.randn(
                (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
            )

        # Pad or truncate to the correct size
        if T >= self.calibration_seq_length:
            inps = inps[: self.calibration_seq_length]
        else:
            inps = F.pad(
                inps, (0, self.calibration_seq_length - T), value=self.pad_token
            )

        inps = inps.unsqueeze(0)
        model_in = self.input_prep_func(inps)

        self.base_input_recorder(*model_in)

        # Output `something` with the correct shape to keep eval going
        return torch.randn(
            (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
        )


InputRecorder = LMEvalInputRecorder  # for BC
