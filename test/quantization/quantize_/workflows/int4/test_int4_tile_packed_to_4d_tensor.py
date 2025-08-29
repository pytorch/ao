# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import (
    Int4TilePackedTo4dTensor,
)
from torchao.quantization.utils import compute_error
from torchao.testing.utils import TorchAOIntegrationTestCase
from torchao.utils import is_sm_at_least_90

INT4_CONFIG = Int4WeightOnlyConfig(
    group_size=128,
    packing_format="tile_packed_to_4d",
    version=2,
)

INT4_HQQ_CONFIG = Int4WeightOnlyConfig(
    group_size=128,
    packing_format="tile_packed_to_4d",
    int4_choose_qparams_algorithm="hqq",
    version=2,
)


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_90(), "Need sm90+")
class TestInt4TilePackedTo4dTensor(TorchAOIntegrationTestCase):
    def setUp(self):
        self.GPU_DEVICES = ["cuda"] if torch.cuda.is_available() else []

    @parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 512, 128),
            ((2, 32, 128), 256, 128),
        ],
    )
    @parametrize("config", [INT4_CONFIG, INT4_HQQ_CONFIG])
    def test_linear(self, sizes, config):
        dtype = torch.bfloat16
        device = "cuda"

        M, N, K = sizes
        input = torch.randn(*M, K, dtype=dtype, device=device)
        linear = torch.nn.Linear(K, N, dtype=dtype, device=device)

        original = linear(input)
        quantize_(linear, config)
        quantized = linear(input)
        self.assertTrue(compute_error(original, quantized) > 20)

        compiled_linear = torch.compile(linear)
        quantized_and_compiled = compiled_linear(input)
        self.assertTrue(compute_error(original, quantized_and_compiled) > 20)

    @parametrize("config", [INT4_CONFIG, INT4_HQQ_CONFIG])
    def test_module_path(self, config):
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        quantize_(linear.cuda(), config)
        self.assertEqual(
            str(type(linear.weight)),
            "<class 'torchao.quantization.Int4TilePackedTo4dTensor'>",
        )

        with tempfile.NamedTemporaryFile() as f:
            torch.save(linear.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)
            self.assertEqual(
                str(type(state_dict["weight"])),
                "<class 'torchao.quantization.Int4TilePackedTo4dTensor'>",
            )

    @parametrize("config", [INT4_CONFIG, INT4_HQQ_CONFIG])
    def test_slice(self, config):
        """Note: we use multiples of 1024 for both in_features and out_features
        so that padding does not affect the weight after slicing
        """
        dtype = torch.bfloat16
        device = "cuda"

        # Create a 2048x2048 linear layer for testing
        dummy = torch.nn.Linear(2048, 2048, bias=False, dtype=dtype, device=device)

        # Create reference sliced linear layers
        dummy1 = torch.nn.Linear(2048, 1024, bias=False, dtype=dtype, device=device)
        dummy1.weight = torch.nn.Parameter(
            dummy.weight.narrow(0, 0, 1024), requires_grad=False
        )
        dummy2 = torch.nn.Linear(1024, 2048, dtype=dtype, device=device)
        dummy2.weight = torch.nn.Parameter(
            dummy.weight.narrow(1, 0, 1024), requires_grad=False
        )

        # Quantize the main linear layer
        quantize_(dummy, config)

        # Shape analysis for TilePackedTo4d format:
        # Original weight shape: (2048, 2048) -> no padding needed (already multiple of 1024)
        # n = 2048, k = 2048, inner_k_tiles = 8, group_size = 128
        #
        # qdata shape: [n/8, k/(inner_k_tiles*16), 32, inner_k_tiles/2]
        #             = [2048/8, 2048/(8*16), 32, 8/2]
        #             = [256, 16, 32, 4]
        #
        # scale_and_zero shape: [in_features/group_size, out_features, 2] (packed format)
        #                     = [2048/128, 2048, 2] = [16, 2048, 2]

        # Test slicing along output dimension (dim=0: 2048 -> 1024)
        weight1 = dummy.weight.narrow(0, 0, 1024)

        # qdata slicing: narrow from [256, 16, 32, 4] to [128, 16, 32, 4]
        # Calculation: 1024 out_features / 2048 total * 256 qdata_dim0 = 128
        expected_qdata_slice_0 = dummy.weight.qdata.narrow(0, 0, 128)
        self.assertEqual(weight1.qdata, expected_qdata_slice_0)

        # scale_and_zero slicing: narrow from [16, 2048, 2] to [16, 1024, 2]
        # slicing 0th dim of qdata means we have to slice 1th dim of scale_and_zero
        expected_scale_zero_slice_0 = dummy.weight.scale_and_zero.narrow(1, 0, 1024)
        self.assertEqual(weight1.scale_and_zero, expected_scale_zero_slice_0)

        # Test slicing along input dimension (dim=1: 2048 -> 1024)
        weight2 = dummy.weight.narrow(1, 0, 1024)

        # qdata slicing: narrow from [256, 16, 32, 4] to [256, 8, 32, 4]
        # k = 2048
        # Calculation: 1024 in_features (1/2 of in_features) corresponds to 1/2 of qdata dimension 1
        # which is k / (inner_k_tiles * 16) / 2 = 2048 / (8 * 16) / 2 = 8
        expected_qdata_slice_1 = dummy.weight.qdata.narrow(1, 0, 8)
        self.assertEqual(weight2.qdata, expected_qdata_slice_1)

        # scale_and_zero slicing: narrow from [16, 2048, 2] to [8, 2048, 2]
        expected_scale_zero_slice_1 = dummy.weight.scale_and_zero.narrow(0, 0, 8)
        self.assertEqual(weight2.scale_and_zero, expected_scale_zero_slice_1)

        # Verify that sliced weights produce similar results to reference implementations
        input1 = torch.randn(2, 2048, dtype=dtype, device=device)
        res_ref1 = dummy1(input1)

        # Create a new linear layer with the sliced weight
        test_linear1 = torch.nn.Linear(
            2048, 1024, bias=False, dtype=dtype, device=device
        )
        test_linear1.weight = torch.nn.Parameter(
            weight1.contiguous(), requires_grad=False
        )
        res1 = test_linear1(input1)
        self.assertGreater(compute_error(res_ref1, res1), 14)

        input2 = torch.randn(2, 1024, dtype=dtype, device=device)
        res_ref2 = dummy2(input2)

        # Create a new linear layer with the sliced weight
        test_linear2 = torch.nn.Linear(
            1024, 2048, bias=False, dtype=dtype, device=device
        )
        test_linear2.weight = torch.nn.Parameter(
            weight2.contiguous(), requires_grad=False
        )
        res2 = test_linear2(input2)
        self.assertGreater(compute_error(res_ref2, res2), 14)

    @parametrize("config", [INT4_CONFIG, INT4_HQQ_CONFIG])
    def test_slice_preserves_aliasing(self, config):
        l = torch.nn.Linear(1024, 1024).to("cuda").to(torch.bfloat16)
        l.weight = torch.nn.Parameter(
            torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cuda")
        )
        quantize_(l, config)
        param = l.weight
        param_data = param.data
        param_data = param_data.narrow(0, 0, 512)
        # Making sure the aliasing is preserved in sliced quantized Tensor
        assert param.data.qdata.data_ptr() == param_data.qdata.data_ptr()
        assert (
            param.data.scale_and_zero.data_ptr() == param_data.scale_and_zero.data_ptr()
        )

    def test_cant_initialize_in_cpu(self):
        config = INT4_CONFIG
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        # make sure there is no cpu implementation of the packing op currently
        with self.assertRaisesRegex(
            NotImplementedError,
            "Could not run 'aten::_convert_weight_to_int4pack' with arguments from the 'CPU' backend. ",
        ):
            quantize_(linear, config)

    def test_to_device(self):
        # test calling to on the tensor that's already on the same device works
        config = INT4_CONFIG

        for device in self.GPU_DEVICES:
            linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=device)
            quantize_(linear, config)
            linear.to(device)

            linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=device)
            quantize_(linear, config)
            linear.to(device=device)

            linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=device)
            quantize_(linear, config)
            linear.to(device)

    @parametrize("config", [INT4_CONFIG, INT4_HQQ_CONFIG])
    def test_slice_and_copy_similar_to_vllm(self, config):
        self._test_slice_and_copy_similar_to_vllm(config)

    @parametrize("device", ["cuda"])
    @parametrize("dtype", [torch.bfloat16])
    def test_mm_int4wo(self, device, dtype):
        weight = torch.randn(512, 1024).to(device).to(dtype)
        weight = weight.t()

        l = torch.nn.Linear(512, 1024).to(device).to(dtype)
        l.weight = torch.nn.Parameter(weight)
        quantize_(l, INT4_CONFIG)
        # weight shape: 1024 x 512
        weight = l.weight

        input = torch.randn(1, 512, device=device, dtype=dtype)
        # make sure it runs
        torch.nn.functional.linear(input, weight)

    @parametrize("group_size", [32, 64, 128])
    def test_different_group_sizes(self, group_size):
        """Test with different group sizes"""
        dtype = torch.bfloat16
        device = "cuda"
        hp_tensor = torch.randn(256, 512, dtype=dtype, device=device)
        block_size = (1, group_size)

        tensor = Int4TilePackedTo4dTensor.from_hp(hp_tensor, block_size)

        self.assertEqual(tensor.shape, hp_tensor.shape)
        self.assertEqual(tensor.block_size, block_size)

    def test_error_conditions(self):
        """Test various error conditions"""
        dtype = torch.bfloat16
        device = "cuda"
        hp_tensor = torch.randn(128, 256, dtype=dtype, device=device)

        # Test invalid block_size length
        with self.assertRaises(AssertionError):
            Int4TilePackedTo4dTensor.from_hp(
                hp_tensor, (64,)
            )  # block_size length mismatch

        # Test non-groupwise quantization
        with self.assertRaises(AssertionError):
            Int4TilePackedTo4dTensor.from_hp(
                hp_tensor, (2, 64)
            )  # first element should be 1


instantiate_parametrized_tests(TestInt4TilePackedTo4dTensor)


if __name__ == "__main__":
    run_tests()
