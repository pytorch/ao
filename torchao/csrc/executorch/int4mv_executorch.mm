#include "metal/int4mv_kernel.h"
#include "metal/mps/MPSStream.h"
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
using namespace torchao::mps;

namespace mps {
  static inline void checkSupportsBFloat16() {
    ET_CHECK_MSG(isMacOS13OrNewer(MacOSVersion::MACOS_VER_14_0_PLUS),
                    "MPS bfloat16 type is supported on MacOS 14.0 or newer.");
  }
  std::string scalarToMetalTypeString(const exec_aten::ScalarType& scalar_type) {
    switch (scalar_type) {
      case ScalarType::Float:
        return "float";
      case ScalarType::Half:
        return "half";
      case ScalarType::BFloat16:
        checkSupportsBFloat16();
        return "bfloat";
      case ScalarType::Int:
        return "int";
      case ScalarType::Long:
        return "long";
      case ScalarType::Short:
        return "short";
      case ScalarType::Char:
        return "char";
      case ScalarType::Byte:
        return "uchar";
      case ScalarType::Bool:
        return "bool";
      default:
        ET_CHECK_MSG(false, "Undefined type %hhd", scalar_type);
        return "Undefined";
    }
  }
} // namespace mps

namespace native {

using RuntimeContext = torch::executor::RuntimeContext;
using ScalarType = torch::executor::ScalarType;
Tensor& _int4mv_out(
  RuntimeContext& ctx,
  const Tensor& A, 
  const Tensor& B,
  int32_t groupSize,
  const Tensor& scalesAndZeros, 
  Tensor& C) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  ET_CHECK_MSG(A.scalar_type() == ScalarType::BFloat16 || A.scalar_type() == ScalarType::Half || A.scalar_type() == ScalarType::Float,
                "%s : expect A to be either 32-bit or 16-bit float tensor.",
                __func__
  );
  ET_CHECK_MSG(A.dim() == 2, "%s : expect A to be 2D tensor.", __func__);

  ET_CHECK_MSG(B.scalar_type() == ScalarType::Char, "%s : expect B to be int8 tensor.", __func__);
  ET_CHECK_MSG(B.size(1) == K, "%s : expect B.size(1) == %zd", __func__, K);

  std::array<uint32_t, 4> sizes = {static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N), 0};
  std::array<uint64_t, 4> nbytes = {A.nbytes(), B.nbytes(), scalesAndZeros.nbytes(), C.nbytes()};
  std::string A_scalar_type = mps::scalarToMetalTypeString(A.scalar_type());
  torchao::int4mv(A.const_data_ptr<uint8_t>(), B.const_data_ptr<uint8_t>(), groupSize, scalesAndZeros.const_data_ptr<uint8_t>(), C.mutable_data_ptr<uint8_t>(), sizes, nbytes, A_scalar_type);
  return C;
}

EXECUTORCH_LIBRARY(
    llama_cpp,
    "_weight_int8pack_mm.out",
    _int4mv_out);
} // namespace native
} // namespace executor
} // namespace torch

