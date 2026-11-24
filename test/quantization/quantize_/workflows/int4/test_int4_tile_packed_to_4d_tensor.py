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

from torchao.quantization import (
    Int4TilePackedTo4dTensor,
    Int4WeightOnlyConfig,
    quantize_,
)
from torchao.quantization.quantize_.common import SupportsActivationPreScaling
from torchao.quantization.utils import compute_error
from torchao.testing.utils import TorchAOIntegrationTestCase
from torchao.utils import is_sm_at_least_90

INT4_CONFIG = Int4WeightOnlyConfig(
    group_size=128,
    int4_packing_format="tile_packed_to_4d",
)

INT4_HQQ_CONFIG = Int4WeightOnlyConfig(
    group_size=128,
    int4_packing_format="tile_packed_to_4d",
    int4_choose_qparams_algorithm="hqq",
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

    @parametrize("config", [INT4_CONFIG, INT4_HQQ_CONFIG])
    def test_activation_prescaling(self, config):
        dtype = torch.bfloat16
        device = "cuda"
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, bias=False, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, config)
        qw = linear.weight
        assert isinstance(qw, SupportsActivationPreScaling), (
            "Expected int4 tensor supports activation prescaling"
        )
        assert qw.act_pre_scale is None, "Default `act_pre_scale` is None"
        _ACT_PRE_SCALE = 2
        qw.act_pre_scale = _ACT_PRE_SCALE
        quantized = linear(input)

        # making sure activation pre scaling is successfully applied to the activation
        self.assertTrue(compute_error(original * _ACT_PRE_SCALE, quantized) > 20)

    @parametrize("config", [INT4_CONFIG, INT4_HQQ_CONFIG])
    def test_select(self, config):
        """Test aten.select.int for MoE expert selection on 3D weight tensors."""
        dtype = torch.bfloat16
        device = "cuda"

        num_experts = 4
        out_features = 1024
        in_features = 2048
        group_size = config.group_size

        # Create a 3D weight tensor simulating MoE experts
        hp_tensor = torch.randn(
            num_experts, out_features, in_features, dtype=dtype, device=device
        )
        block_size = [1, 1, group_size]

        # Quantize the 3D tensor
        qt = Int4TilePackedTo4dTensor.from_hp(hp_tensor, block_size)

        # Select expert at index 0
        selected_qt = torch.select(qt, 0, 0)
        selected_hp = Int4TilePackedTo4dTensor.from_hp(
            torch.select(hp_tensor, 0, 0), [1, group_size]
        )

        # Verify shape is reduced from 3D to 2D
        self.assertEqual(selected_qt.shape, selected_hp.shape)
        self.assertEqual(selected_qt.block_size, selected_hp.block_size)

        # Verify qdata and scale_and_zero are correctly indexed
        torch.testing.assert_close(selected_qt.qdata, selected_hp.qdata, atol=0, rtol=0)
        torch.testing.assert_close(
            selected_qt.scale_and_zero, selected_hp.scale_and_zero, atol=0, rtol=0
        )

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


class TestPaddedKSlices(unittest.TestCase):
    def test_packed_storage(self):
        n, k, padded_k = 16, 9728, 10240
        group_size = 64
        qdata = torch.arange(n * padded_k // 8, dtype=torch.int32).view(
            n // 8, padded_k // 128, 32, 4
        )
        scales = torch.arange(padded_k // group_size * n * 2).view(-1, n, 2).bfloat16()
        weight = Int4TilePackedTo4dTensor(
            qdata, scales, [1, group_size], (n, k), torch.arange(k)
        )
        shard_k = k // 2
        dst = Int4TilePackedTo4dTensor(
            torch.ones_like(qdata[:, :40]),
            torch.ones_like(scales[:80]),
            [1, group_size],
            (n, shard_k),
            torch.zeros(shard_k, dtype=torch.int64),
        )
        for start in (0, shard_k):
            shard = weight.narrow(1, start, shard_k)
            end = start + shard_k
            expected_qdata = qdata[:, start // 128 : end // 128]
            expected_scales = scales[start // group_size : end // group_size]
            torch.testing.assert_close(shard.qdata, expected_qdata)
            torch.testing.assert_close(shard.scale_and_zero, expected_scales)
            self.assertEqual(shard.qdata.data_ptr(), expected_qdata.data_ptr())
            self.assertEqual(
                shard.scale_and_zero.data_ptr(), expected_scales.data_ptr()
            )
            dst.copy_(shard)
            torch.testing.assert_close(dst.qdata[:, :38], expected_qdata)
            torch.testing.assert_close(dst.scale_and_zero[:76], expected_scales)
            self.assertEqual(dst.qdata[:, 38:].count_nonzero().item(), 0)
            self.assertEqual(dst.scale_and_zero[76:].count_nonzero().item(), 0)
            torch.testing.assert_close(
                dst.act_pre_scale, weight.act_pre_scale[start:end]
            )
            dst.qdata[:, 0].fill_(-1)
            shard.copy_(dst)
            self.assertTrue((qdata[:, start // 128] == -1).all())

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        "Requires the CUDA int4 tensor core kernel",
    )
    def test_tensor_parallel_shards(self):
        torch.manual_seed(0)
        for k in (3072, 9728):
            with self.subTest(k=k):
                n, shard_k = 128, k // 2
                weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                packed = Int4TilePackedTo4dTensor.from_hp(weight, [1, 128])
                packed.act_pre_scale = torch.linspace(
                    0.5, 1.5, k, device="cuda", dtype=weight.dtype
                )
                x = torch.randn(4, k, device="cuda", dtype=weight.dtype)
                dst = Int4TilePackedTo4dTensor.from_hp(weight[:, :shard_k], [1, 128])
                dst.act_pre_scale = torch.empty_like(packed.act_pre_scale[:shard_k])
                for start in (0, shard_k):
                    shard = packed.narrow(1, start, shard_k)
                    self.assertEqual(
                        shard.qdata.untyped_storage().data_ptr(),
                        packed.qdata.untyped_storage().data_ptr(),
                    )
                    masked = torch.zeros_like(x)
                    masked[:, start : start + shard_k] = x[:, start : start + shard_k]
                    expected = torch.nn.functional.linear(masked, packed)
                    local_x = x[:, start : start + shard_k].contiguous()
                    dst.copy_(shard)
                    for local_weight in (shard, dst):
                        torch.testing.assert_close(
                            torch.nn.functional.linear(local_x, local_weight),
                            expected,
                            rtol=0.016,
                            atol=0.016,
                        )


if __name__ == "__main__":
    run_tests()
