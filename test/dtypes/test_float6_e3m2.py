# import torch
# from torch.testing._internal.common_utils import (
#     TestCase,
#     instantiate_parametrized_tests,
#     parametrize,
#     run_tests,
# )

# try:
#     import torchao.ops
# except RuntimeError:
#     pytest.skip("torchao.ops not available")


# from torchao.dtypes.float6_e3m2 import to_float6_e3m2, from_float6_e3m2


# _DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# _DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# class TestFloat6E3M2(TestCase):

#     @parametrize("device", _DEVICES)
#     @parametrize("dtype", _DTYPES)
#     @parametrize(
#         "input_output",
#         [
#             (0.0,    0b000000),  # exact values
#             (1.0,    0b001100),  # normal numbers
#             (1.25,   0b001101),
#             (28.0,   0b011111),  # max
#             (0.1875, 0b000011),  # subnormal number
#             (0.0625, 0b000001),  # min
#             (29.0,   0b011111),  # normal round down
#             (26.0,   0b011110),  # normal round to nearest even
#             (0.1251, 0b000010),  # subnormal round down
#             (0.0314, 0b000001),  # subnormal round up
#             (0.03,   0b000000),  # underflow
#         ],
#     )
#     def test_to_float6_e3m2_no_bit_packing_correctness(self, device, dtype, input_output):
#         input, output = input_output
#         input = torch.tensor(input, device=device, dtype=dtype)
#         assert to_float6_e3m2(input, no_bit_packing=True).item() == output

#     @parametrize("device", _DEVICES)
#     @parametrize("dtype", _DTYPES)
#     def test_to_float6_e3m2_bit_packing_correctness(self, device, dtype):
#         x = torch.randn(128, 128, device=device, dtype=dtype)
#         results_unpacked = to_float6_e3m2(x, no_bit_packing=True)
#         results_packed = to_float6_e3m2(x)

#         val0, val1, val2, val3 = results_unpacked.unflatten(-1, (-1, 4)).unbind(-1)
#         bits0 = (val0 << 2) | (val1 >> 4)  # 0000 0011
#         bits1 = (val1 << 4) | (val2 >> 2)  # 1111 2222
#         bits2 = (val2 << 6) | (val3);      # 2233 3333

#         expected_packed = torch.stack([bits0, bits1, bits2], dim=-1).flatten(-2)
#         assert (results_packed == expected_packed).all()

#     @parametrize("device", _DEVICES)
#     @parametrize("shape", [(), (0,), (10,), (20, 20)])
#     def test_to_float6_e3m2_no_bit_packing_shape(self, device, shape):
#         x = torch.randn(shape, device=device)
#         result = to_float6_e3m2(x, no_bit_packing=True)
#         assert result.shape == shape

#     @parametrize("device", _DEVICES)
#     @parametrize("shape", [(4,), (20, 20)])
#     def test_to_float6_e3m2_bit_packing_shape(self, device, shape):
#         x = torch.randn(shape, device=device)
#         result = to_float6_e3m2(x)
#         assert result.shape == shape[:-1] + (shape[-1] // 4 * 3,)

#     @parametrize("device", _DEVICES)
#     @parametrize("dtype", _DTYPES)
#     @parametrize("no_bit_packing", [False, True])
#     def test_to_float6_e3m2_compile(self, device, dtype, no_bit_packing):
#         x = torch.randn(20, 20, device=device, dtype=dtype)
#         expected = to_float6_e3m2(x, no_bit_packing=no_bit_packing)

#         to_float6_e3m2_compiled = torch.compile(to_float6_e3m2)
#         actual = to_float6_e3m2_compiled(x, no_bit_packing=no_bit_packing)
#         torch.testing.assert_close(actual, expected)

#     @parametrize("device", _DEVICES)
#     @parametrize(
#         "input_output",
#         [
#             (0b000000, 0.0),
#             (0b001100, 1.0),
#             (0b011111, 28.0),    # max
#             (0b000001, 0.0625),  # min
#             (0b001110, 1.5),
#             (0b000011, 0.1875),  # subnormal
#         ],
#     )
#     def test_from_float6_e3m2_no_bit_packing_correctness(self, device, input_output):
#         input, output = input_output
#         input = torch.tensor(input, device=device, dtype=torch.uint8)
#         assert from_float6_e3m2(input, no_bit_packing=True).item() == output

#     @parametrize("device", _DEVICES)
#     def test_from_float6_e3m2_bit_packing_correctness(self, device):
#         x = torch.randint(256, (128, 128 // 4 * 3), device=device, dtype=torch.uint8)
#         actual = from_float6_e3m2(x)

#         bits0, bits1, bits2 = x.unflatten(-1, (-1, 3)).unbind(-1)
#         x_unpacked0 = bits0 >> 2
#         x_unpacked1 = ((bits0 & 0x3) << 4) | (bits1 >> 4)
#         x_unpacked2 = ((bits1 & 0xF) << 2) | (bits2 >> 6)
#         x_unpacked3 = bits2 & 0x3F

#         x_unpacked = torch.stack([x_unpacked0, x_unpacked1, x_unpacked2, x_unpacked3], dim=-1).flatten(-2)
#         expected = from_float6_e3m2(x_unpacked, no_bit_packing=True)
#         torch.testing.assert_close(actual, expected)

#     @parametrize("device", _DEVICES)
#     @parametrize("no_bit_packing", [False, True])
#     def test_from_float6_e3m2_compile(self, device, no_bit_packing):
#         x = torch.randint(256, size=(20, 15), device=device, dtype=torch.uint8)
#         expected = from_float6_e3m2(x, no_bit_packing=no_bit_packing)

#         from_float6_e3m2_compiled = torch.compile(from_float6_e3m2)
#         actual = from_float6_e3m2_compiled(x, no_bit_packing=no_bit_packing)
#         torch.testing.assert_close(actual, expected)


# instantiate_parametrized_tests(TestFloat6E3M2)


# if __name__ == "__main__":
#     run_tests()
