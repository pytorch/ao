set -eu

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <build_only|quantization|bitpacking|linear>";
    exit 1;
fi

BENCHMARK_TYPE="${1}"

export CMAKE_OUT=cmake-out

export CMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')
echo "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}"

# Build
cmake -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_OUT} \
    -DTORCHAO_BUILD_EXECUTORCH_OPS=OFF \
    -DTORCHAO_BUILD_CPU_AARCH64=ON \
    -DTORCHAO_ENABLE_ARM_NEON_DOT=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DTORCHAO_BUILD_TESTS=OFF \
    -DTORCHAO_BUILD_BENCHMARKS=ON \
    -DOpenMP_ROOT=$(brew --prefix libomp) \
    -S . \
    -B ${CMAKE_OUT}
cmake --build  ${CMAKE_OUT} -j 16 --config Release


# Run
TARGET_PREFIX="${CMAKE_OUT}/torch_free_kernels/aarch64/benchmarks/torchao_benchmarks_torch_free_kernels_aarch64_"
case "${BENCHMARK_TYPE}" in
    build_only) echo "Build only"; exit 0; ;;
    quantization) ${TARGET_PREFIX}benchmark_quantization; ;;
    bitpacking) ${TARGET_PREFIX}benchmark_bitpacking; ;;
    linear) ${TARGET_PREFIX}benchmark_linear; ;;
    *) echo "Unknown benchmark: $1. Please specify quantization, bitpacking, or linear."; exit 1; ;;
esac
