#include <hip/hip_runtime.h>

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/hip/HIPBlas.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

using at::Scalar;
using at::Tensor;
using at::TensorArg;
using c10::IntArrayRef;

//
// copied from aten/src/ATen/cuda/CUDABlas.cpp
//
namespace {

static hipblasOperation_t _cublasOpFromChar(char op) {
  // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
  switch (op) {
    case 'n':
    case 'N':
      return HIPBLAS_OP_N;
    case 't':
    case 'T':
      return HIPBLAS_OP_T;
    case 'c':
    case 'C':
      return HIPBLAS_OP_C;
  }
  TORCH_CHECK(false,
      "_cublasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
}

static void _cublasAdjustLdLevel3(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc) {
  bool transa_ = ((transa != 'n') && (transa != 'N'));
  bool transb_ = ((transb != 'n') && (transb != 'N'));

  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).
  if (n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if (transa_) {
    if (m <= 1)
      *lda = std::max<int64_t>(k, 1);
  } else {
    if (k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if (transb_) {
    if (k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  } else {
    if (n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }
}

// Following the pattern of CuSparseDescriptor
// Defined here for now because this is the only place cublas_lt interface is
// used but can be moved to a header once cublas_lt interface is used in
// multiple places.
template <typename T, hipblasStatus_t (*destructor)(T*)>
struct HipBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CUDABLAS_CHECK(destructor(x));
    }
  }
};

template <typename T, hipblasStatus_t (*destructor)(T*)>
class HipBlasLtDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, HipBlasLtDeleter<T, destructor>> descriptor_;
};

class HipBlasLtMatmulDescriptor : public HipBlasLtDescriptor<
                                     hipblasLtMatmulDescOpaque_t,
                                     &hipblasLtMatmulDescDestroy> {
 public:
  HipBlasLtMatmulDescriptor(
      hipblasComputeType_t compute_type,
      hipDataType scale_type) {
    hipblasLtMatmulDesc_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(
        hipblasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(hipblasLtMatmulDescAttributes_t attr, const T value) {
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    TORCH_CUDABLAS_CHECK(::hipblasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(value)));
  }
};

class HipBlasLtMatrixLayout : public HipBlasLtDescriptor<
                                 hipblasLtMatrixLayoutOpaque_t,
                                 &hipblasLtMatrixLayoutDestroy> {
 public:
  HipBlasLtMatrixLayout(
      hipDataType type,
      uint64_t rows,
      uint64_t cols,
      int64_t ld,
      bool t = false) {
    hipblasLtMatrixLayout_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(
        hipblasLtMatrixLayoutCreate(&raw_descriptor, type, t ? cols : rows, t ? rows : cols, ld));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(hipblasLtMatrixLayoutAttribute_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(::hipblasLtMatrixLayoutSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

class HipBlasLtMatmulPreference : public HipBlasLtDescriptor<
                                     hipblasLtMatmulPreferenceOpaque_t,
                                     &hipblasLtMatmulPreferenceDestroy> {
 public:
  HipBlasLtMatmulPreference() {
    hipblasLtMatmulPreference_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(hipblasLtMatmulPreferenceCreate(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(hipblasLtMatmulPreferenceAttributes_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(::hipblasLtMatmulPreferenceSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

static size_t _parseChosenWorkspaceSize() {
  auto val = c10::utils::get_env("CUBLASLT_WORKSPACE_SIZE");
#ifdef USE_ROCM
  if (!val.has_value()) {
    // accept either env var
    val = c10::utils::get_env("HIPBLASLT_WORKSPACE_SIZE");
  }
  size_t workspace_size = 76*1024; /* Use 76 MB for hipBLASLt */
#else
  size_t workspace_size = 1024; /* default size in KiB according to #73328 */
#endif

  if (val.has_value()) {
    try {
      workspace_size = std::stoi(val.value());
    } catch(std::invalid_argument const& e) {
      TORCH_WARN("invalid CUBLASLT_WORKSPACE_SIZE,",
                 " using default workspace size of ", workspace_size, " KiB.");
    } catch(std::out_of_range const& e) {
      TORCH_WARN("CUBLASLT_WORKSPACE_SIZE out of range,",
                 " using default workspace size of ", workspace_size, " KiB.");
    }
  }
  return workspace_size * 1024;
}

static size_t _getWorkspaceSize() {
  static size_t workspace_size = _parseChosenWorkspaceSize();
  return workspace_size;
}

} // namespace

//
// copied from aten/src/ATen/native/cuda/Blas.cpp
//
namespace {

// TODO: https://github.com/pytorch/pytorch/pull/59380#pullrequestreview-725310492
c10::MaybeOwned<Tensor> inline resolve_conj_if_indicated(const Tensor& tensor, bool resolve_conj) {
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor, bool transpose_result) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return resolve_conj_if_indicated(tensor, transpose_result ? transpose_tensor : !transpose_tensor);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, !transpose_result);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, transpose_result);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return resolve_conj_if_indicated(tensor, true);
  }

  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, true);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, true);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}

struct cublasCommonArgs {
  cublasCommonArgs(const Tensor& mat1, const Tensor& mat2, Tensor& c) {
    bool transpose_result = false, transpose_mat1 = false, transpose_mat2 = false;
    result = prepare_matrix_for_cublas(c, transpose_result);
    mata = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_mat1, transpose_result);
    matb = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_mat2, transpose_result);
    auto mat1_sizes = mat1.sizes();
    auto mat2_sizes = mat2.sizes();
    if (transpose_result) {
      transpose_mat1 = !transpose_mat1;
      transpose_mat2 = !transpose_mat2;
      mat1_sizes = mata->sizes();
      mat2_sizes = matb->sizes();
    }

    m = mat1_sizes[transpose_result ? 1 : 0];
    k = mat1_sizes[transpose_result ? 0 : 1];
    n = mat2_sizes[transpose_result ? 0 : 1];
    lda = mata->stride((transpose_mat1 == transpose_result) ? 1 : 0);
    ldb = matb->stride((transpose_mat2 == transpose_result) ? 1 : 0);
    result_ld = result->stride(transpose_result ? 0 : 1);
    transa = transpose_mat1 ?  mata->is_conj() ? 'c' : 't' : 'n';
    transb = transpose_mat2 ?  matb->is_conj() ? 'c' : 't' : 'n';
  }
  char transa, transb;
  int64_t m, n, k;
  int64_t lda, ldb, result_ld;
  c10::MaybeOwned<Tensor> mata, matb, result;
};

} // namespace

template <typename Dtype>
inline void bgemm_hipblaslt(CUDABLAS_BGEMM_ARGTYPES(Dtype), bool mat1_is_swizzled, bool mat2_is_swizzled) {
  hipDataType abcType = HIP_R_32F;
  hipblasComputeType_t computeType = HIPBLAS_COMPUTE_32F;
  hipDataType scaleType = HIP_R_32F;
  if constexpr (std::is_same_v<Dtype, double>) {
    abcType = HIP_R_64F;
    computeType = HIPBLAS_COMPUTE_64F;
    scaleType = HIP_R_64F;
  } else if constexpr (std::is_same_v<Dtype, float>) {
  } else if constexpr (std::is_same_v<Dtype, c10::complex<double>>) {
    abcType = HIP_C_64F;
    computeType = HIPBLAS_COMPUTE_64F;
    scaleType = HIP_C_64F;
  } else if constexpr (std::is_same_v<Dtype, c10::complex<float>>) {
    abcType = HIP_C_32F;
    scaleType = HIP_C_32F;
  } else if constexpr (std::is_same_v<Dtype, at::Half>) {
    abcType = HIP_R_16F;
  } else if constexpr (std::is_same_v<Dtype, at::BFloat16>) {
    abcType = HIP_R_16BF;
  } else {
    static_assert(false && sizeof(Dtype), "at::cuda::blas::bgemm_internal_cublaslt: not implemented");
  }

  hipblasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
  hipblasOperation_t opa = _cublasOpFromChar(transa);
  hipblasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

  HipBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  computeDesc.setAttribute(HIPBLASLT_MATMUL_DESC_TRANSA, opa);
  computeDesc.setAttribute(HIPBLASLT_MATMUL_DESC_TRANSB, opb);
  HipBlasLtMatrixLayout Adesc(abcType, m, k, lda, opa == HIPBLAS_OP_T);
  HipBlasLtMatrixLayout Bdesc(abcType, k, n, ldb, opb == HIPBLAS_OP_T);
  HipBlasLtMatrixLayout Cdesc(abcType, m, n, ldc);
#ifdef HIPBLASLT_HAS_ORDER_COL16
  if (mat1_is_swizzled) {
    Adesc.setAttribute(HIPBLASLT_MATRIX_LAYOUT_ORDER, HIPBLASLT_ORDER_COL16_4R8);
  }
  if (mat2_is_swizzled) {
    Bdesc.setAttribute(HIPBLASLT_MATRIX_LAYOUT_ORDER, HIPBLASLT_ORDER_COL16_4R8);
  }
#endif

  if (num_batches > 1) {
    int num_batches_as_int = static_cast<int>(num_batches);
    Adesc.setAttribute(HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Bdesc.setAttribute(HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Cdesc.setAttribute(HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Adesc.setAttribute(HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stridea);
    Bdesc.setAttribute(HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, strideb);
    Cdesc.setAttribute(HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stridec);
  }

  HipBlasLtMatmulPreference preference;
  // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind
  // setting this to 1M.
  size_t workspaceSize = _getWorkspaceSize();
  preference.setAttribute(HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);

#ifndef USE_ROCM
  uint32_t a_alignment = _getAlignment(reinterpret_cast<uintptr_t>(a));
  uint32_t b_alignment = _getAlignment(reinterpret_cast<uintptr_t>(b));
  uint32_t c_alignment = _getAlignment(reinterpret_cast<uintptr_t>(c));
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);
#endif

  auto workspace = at::empty(static_cast<int64_t>(workspaceSize), at::TensorOptions().dtype(at::kByte).device(at::kCUDA));

  hipblasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  TORCH_CUDABLAS_CHECK(hipblasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Cdesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  if (returnedResult == 0) {
    TORCH_CUDABLAS_CHECK(HIPBLAS_STATUS_NOT_SUPPORTED);
  }

  hipblasStatus_t cublasStatus = hipblasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      &alpha,
      a,
      Adesc.descriptor(),
      b,
      Bdesc.descriptor(),
      &beta,
      c,
      Cdesc.descriptor(),
      c,
      Cdesc.descriptor(),
      &heuristicResult.algo,
      workspace.mutable_data_ptr(),
      workspaceSize,
      at::hip::getCurrentHIPStreamMasqueradingAsCUDA());
  TORCH_CHECK(
      cublasStatus == HIPBLAS_STATUS_SUCCESS,
      "CUDA error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling hipblasLtMatmul with transpose_mat1 ",
      (opa == HIPBLAS_OP_T),
      " transpose_mat2 ",
      (opb == HIPBLAS_OP_T),
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " lda ",
      lda,
      " ldb ",
      ldb,
      " ldc ",
      ldc,
      " abcType ",
      abcType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
}


template <typename Dtype>
inline void gemm_hipblaslt(CUDABLAS_GEMM_ARGTYPES(Dtype), bool mat1_is_swizzled, bool mat2_is_swizzled) {
  // forward to bgemm implementation but set strides and batches to 0
  bgemm_hipblaslt(transa, transb, m, n, k, alpha, a, lda, 0, b, ldb, 0, beta, c, ldc, 0, 0, mat1_is_swizzled, mat2_is_swizzled);
}


Tensor swizzle_mm(const Tensor& mat1, const Tensor& mat2, bool mat1_is_swizzled, bool mat2_is_swizzled) {
  TORCH_CHECK(
    mat1.dtype() == mat2.dtype(),
    "expected mat1 and mat2 to have the same dtype, but got: ", mat1.dtype(), " != ", mat2.dtype()
  );

  // NOLINTNEXTLINE(*c-array*)
  TensorArg targs[]{{mat1, "mat1", 0}, {mat2, "mat2", 1}};
  checkAllSameGPU(__func__, targs);

  Tensor meta_mat1 = mat1.to("meta");
  Tensor meta_mat2 = mat2.to("meta");
  Tensor meta_result = at::mm(meta_mat1, meta_mat2);
  Tensor result = at::empty_like(meta_result, mat1.device());
  at::ScalarType scalar_type = result.scalar_type();

  cublasCommonArgs args(mat1, mat2, result);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scalar_type,
      "addmm_cuda",
      [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        opmath_t alpha_val = opmath_t(1.0);
        opmath_t beta_val = opmath_t(0.0);
        const scalar_t* mat1_ptr = args.mata->const_data_ptr<scalar_t>();
        const scalar_t* mat2_ptr = args.matb->const_data_ptr<scalar_t>();
        scalar_t* result_ptr = args.result->mutable_data_ptr<scalar_t>();
        gemm_hipblaslt<scalar_t>(
            args.transa,
            args.transb,
            args.m,
            args.n,
            args.k,
            alpha_val,
            mat1_ptr,
            args.lda,
            mat2_ptr,
            args.ldb,
            beta_val,
            result_ptr,
            args.result_ld,
            mat1_is_swizzled,
            mat2_is_swizzled);
      });

    return result;
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::swizzle_mm", &swizzle_mm);
}
