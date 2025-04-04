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

from torchao.quantization.GPTQ_MT import MultiTensor
from torchao.quantization.utils import _MultiInput

try:  # lm_eval version 0.4
    from lm_eval.evaluator import evaluate  # pyre-ignore[21]
    from lm_eval.models.huggingface import HFLM as eval_wrapper  # pyre-ignore[21]
    from lm_eval.tasks import get_task_dict  # pyre-ignore[21]
except:  # lm_eval version 0.3
    from lm_eval import base, evaluator, tasks

    eval_wrapper = base.BaseLM
    get_task_dict = tasks.get_task_dict
    evaluate = evaluator.evaluate


class MultiTensorInputRecorder(eval_wrapper):
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

        # Initialize inputs as a list of two empty lists for input tensors and indices
        self.inputs = [[], []]

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
        # Ensure that inputs are added correctly as pairs
        self.inputs[0].append(args[0])
        self.inputs[1].append(args[1])

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

    def get_inputs(self):
        # Return MultiTensor instances for both inputs and indices
        return [MultiTensor(self.inputs[0]), MultiTensor(self.inputs[1])]

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

        self.add_input(model_in)

        # Output `something` with the correct shape to keep eval going
        return torch.randn(
            (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
        )

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception("unimplemented")


class InputRecorder(eval_wrapper):
    """
    This is a fake evaluation wrapper from the lm_eval library that just records the inputs
    so that they can be used in calibration.

    If pad_calibration_inputs is enabled, the input recorder will take
    each input and pad/truncate it down to the calibration_seq_length.
    (if using padding you should set the embeddings for the pad_token to 0
    in the model)

    Note: after padding/truncation, input_prep_function is called to bring
    it to the proper form to be inserted into a given model.

    If not, it will only truncate inputs to the desired length.
    """

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

        # need to take inps and convert to corrent input
        # for model
        self.input_prep_func = (
            input_prep_func if input_prep_func is not None else lambda x: (x,)
        )

        self.pad_calibration_inputs = pad_calibration_inputs
        self.pad_token = pad_token

        self.inputs = None

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
        # TODO: verify this for multi-batch as well
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
        if self.inputs is None:
            self.inputs = [_MultiInput([arg]) for arg in args]
        else:
            self.inputs = [
                multi.add_input(arg) for (multi, arg) in zip(self.inputs, args)
            ]

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
        # TODO: make batches work
        input = self.input_prep_func(inps)

        max_seq_length = min(max(inps.size()), self.max_length)
        with torch.device(self._device):
            self._model.setup_caches(self.batch_size, max_seq_length)
        logits = self._model(*input)
        return logits

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception("unimplemented")

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
