# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAMES = [
    "torchao-testing/opt-125m-float8dq-row-fbgemm",
]


class TestSerializationBC(TestCase):
    """Test we can still load and run serialized model in previous AO versions
    we commit to have BC for 3 pytorch releases
    """

    @common_utils.parametrize("model_name", _MODEL_NAMES)
    def test_load_model_and_run(self, model_name):
        # Load and quantize model
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="bfloat16",
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        prompt = ("Hello, my name is",)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")
        generated_ids = quantized_model.generate(**inputs, max_new_tokens=128)
        # make sure it runs
        _ = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


common_utils.instantiate_parametrized_tests(TestSerializationBC)

if __name__ == "__main__":
    run_tests()
