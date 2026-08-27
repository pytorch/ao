#!/usr/bin/env bash
set -euo pipefail

# The CPU PT2E workflow owns the full suite. Keep this list limited to tests
# that exercise accelerator-specific paths.
pt2e_gpu_tests=(
  test/quantization/pt2e/test_quantize_pt2e_qat.py::TestQuantizePT2EQAT_ConvBn1d::test_qat_conv_bn_fusion_cuda
  test/quantization/pt2e/test_quantize_pt2e_qat.py::TestQuantizePT2EQAT_ConvBn1d::test_qat_conv_bn_relu_fusion_cuda
  test/quantization/pt2e/test_quantize_pt2e_qat.py::TestQuantizePT2EQAT_ConvBn2d::test_qat_conv_bn_fusion_cuda
  test/quantization/pt2e/test_quantize_pt2e_qat.py::TestQuantizePT2EQAT_ConvBn2d::test_qat_conv_bn_relu_fusion_cuda
  test/quantization/pt2e/test_quantize_pt2e.py::TestQuantizePT2E::test_move_exported_model_bn_device_cuda
  test/quantization/pt2e/test_quantize_pt2e.py::TestQuantizePT2E::test_allow_exported_model_train_eval
  test/quantization/pt2e/test_learnable_fake_quantize.py::TestLearnableFakeQuantizeIntegration::test_device_compatibility
  test/quantization/pt2e/test_learnable_fake_quantize.py::TestLearnableFakeQuantizeComparison::test_numerical_consistency_per_tensor
)

pytest "${pt2e_gpu_tests[@]}" --verbose -s --durations=25 "$@"
