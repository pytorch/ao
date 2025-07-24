// setup.py glob includes all *.cpp files
// but only build this for ROCm
#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/hip/HIPDataType.h>
#include <ATen/hip/HIPBlas.h>
#include <ATen/native/Resize.h>
#include <c10/core/ScalarType.h>
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

using at::Scalar;
using at::Tensor;
using at::TensorArg;
using c10::kFloat;
using c10::ScalarType;
using c10::IntArrayRef;
using at::cuda::ScalarTypeToCudaDataType;

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
  if (!val.has_value()) {
    // accept either env var
    val = c10::utils::get_env("HIPBLASLT_WORKSPACE_SIZE");
  }
  size_t workspace_size = 76*1024; /* Use 76 MB for hipBLASLt */

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

static bool _scaled_mm_is_fnuz() {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    std::string device_arch = dprops->gcnArchName;
    static const std::vector<std::string> archs = {"gfx940", "gfx941", "gfx942"};
    for (std::string arch : archs) {
        size_t substring = device_arch.find(arch);
        if (substring != std::string::npos) {
            return true;
        }
    }
    return false;
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
  cublasCommonArgs(
      const Tensor& mat1,
      const Tensor& mat2,
      bool swizzle1,
      bool swizzle2,
      Tensor& c,
      const std::optional<Tensor>& scale_a = std::nullopt,
      const std::optional<Tensor>& scale_b = std::nullopt,
      const std::optional<Tensor>& scale_result = std::nullopt) {
    bool transpose_result = false, transpose_a = false, transpose_b = false;
    result = prepare_matrix_for_cublas(c, transpose_result);
    mata = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_a, transpose_result);
    matb = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_b, transpose_result);

    // Handle scale tensors if provided
    if (scale_a && scale_b) {
      // By default since we return in row-major we run the gemm
      // as B.T @ A.T, check transpose_result to determine if we flip the scales
      scale_mata_ptr = transpose_result ? scale_b->data_ptr() : scale_a->data_ptr();
      scale_mata_dtype = transpose_result ? scale_b->scalar_type() : scale_a->scalar_type();
      scale_matb_ptr = transpose_result ? scale_a->data_ptr() : scale_b->data_ptr();
      scale_matb_dtype = transpose_result ? scale_a->scalar_type() : scale_b->scalar_type();
    }

    if (scale_result) {
      scale_result_ptr = scale_result->data_ptr();
      scale_result_dtype = scale_result->scalar_type();
    }

    // Update transpose flags
    if (transpose_result) {
      transpose_a = !transpose_a;
      transpose_b = !transpose_b;
    }

    auto sizes_a = mata->sizes();
    auto sizes_b = matb->sizes();

    m = sizes_a[transpose_result ? 1 : 0];
    k = sizes_a[transpose_result ? 0 : 1];
    n = sizes_b[transpose_result ? 0 : 1];
    lda = mata->stride((transpose_a == transpose_result) ? 1 : 0);
    ldb = matb->stride((transpose_b == transpose_result) ? 1 : 0);
    result_ld = result->stride(transpose_result ? 0 : 1);
    transa = transpose_a ? mata->is_conj() ? 'c' : 't' : 'n';
    transb = transpose_b ? matb->is_conj() ? 'c' : 't' : 'n';

    mata_is_swizzled = transpose_result ? swizzle2 : swizzle1;
    matb_is_swizzled = transpose_result ? swizzle1 : swizzle2;
  }

  // Matrix members
  char transa, transb;
  int64_t m, n, k;
  int64_t lda, ldb, result_ld;
  c10::MaybeOwned<Tensor> mata, matb, result;

  // Scale members
  void* scale_mata_ptr = nullptr;
  void* scale_matb_ptr = nullptr;
  void* scale_result_ptr = nullptr;
  std::optional<c10::ScalarType> scale_mata_dtype;
  std::optional<c10::ScalarType> scale_matb_dtype;
  std::optional<c10::ScalarType> scale_result_dtype;

  // swizzle members
  bool mata_is_swizzled;
  bool matb_is_swizzled;
};

enum class ScalingType {
  TensorWise,
  RowWise,
  Error
};

ScalingType get_scaling_type(
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    int64_t dim_m,
    int64_t dim_n) {
  // Both Per-Tensor and Row-wise scaling expect fp32 tensors
  TORCH_CHECK(
      scale_a.scalar_type() == kFloat && scale_b.scalar_type() == kFloat,
      "Both scale_a and scale_b must be float (fp32) tensors.");

  // Check the singluar scale case for per-tensor scaling
  if (scale_a.numel() == 1 && scale_b.numel() == 1) {
    return ScalingType::TensorWise;
  }

  // For non-TensorWise scaling, enforce 2D input tensors
  TORCH_CHECK(
      scale_a.dim() == 2 && scale_b.dim() == 2,
      "For non-TensorWise scaling, scale tensors must be 2-dimensional, "
      "but got scale_a.dim()=",
      scale_a.dim(),
      " and scale_b.dim()=",
      scale_b.dim());

  // Check for RowWise scaling
  if (scale_a.size(0) == dim_m && scale_a.size(1) == 1 &&
      scale_b.size(0) == 1 && scale_b.size(1) == dim_n) {
#if defined(HIPBLASLT_VEC_EXT)
    TORCH_CHECK(
        scale_a.is_contiguous() && scale_b.is_contiguous(),
        "Both scale_a and scale_b must be contiguous for RowWise scaling.");
    return ScalingType::RowWise;
#else
    TORCH_CHECK(false, "Per-row scaling is not supported for this platform!");
    return ScalingType::Error;
#endif
  }

  // If we reach here, the input doesn't match any valid scaling type
  TORCH_CHECK(
      false,
      "Invalid scaling configuration. For TensorWise scaling, both scales should be scalar. "
      "For RowWise scaling, scale_a should be (",
      dim_m,
      ", 1) and scale_b should be (1, ",
      dim_n,
      "). "
      "Got scale_a.size()=(",
      scale_a.size(0),
      ", ",
      scale_a.size(1),
      ") and ",
      "scale_b.size()=(",
      scale_b.size(0),
      ", ",
      scale_b.size(1),
      ")");

  return ScalingType::Error;
}

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

  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
  computeDesc.setAttribute(HIPBLASLT_MATMUL_DESC_EPILOGUE, epilogue);

  HipBlasLtMatmulPreference preference;
  // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind
  // setting this to 1M.
  size_t workspaceSize = _getWorkspaceSize();
  preference.setAttribute(HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);

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

  cublasCommonArgs args(mat1, mat2, mat1_is_swizzled, mat2_is_swizzled, result);

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
            args.mata_is_swizzled,
            args.matb_is_swizzled);
      });

    return result;
}

void _scaled_gemm(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    const void* mat1_ptr,
    const void* mat1_scale_ptr,
    int64_t mat1_ld,
    ScalarType mat1_dtype,
    ScalarType mat1_scale_dtype,
    bool mat1_is_swizzled,
    const void* mat2_ptr,
    const void* mat2_scale_ptr,
    int64_t mat2_ld,
    ScalarType mat2_dtype,
    ScalarType mat2_scale_dtype,
    bool mat2_is_swizzled,
    const void* bias_ptr,
    ScalarType bias_dtype,
    void* result_ptr,
    const void *result_scale_ptr,
    int64_t result_ld,
    ScalarType result_dtype,
    bool use_rowwise) {
  const auto computeType = HIPBLAS_COMPUTE_32F;
  const auto scaleType = HIP_R_32F;
  const float alpha_val = 1.0;
  const float beta_val = 0.0;
  HipBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  computeDesc.setAttribute(HIPBLASLT_MATMUL_DESC_TRANSA, _cublasOpFromChar(transa));
  computeDesc.setAttribute(HIPBLASLT_MATMUL_DESC_TRANSB, _cublasOpFromChar(transb));
  hipblasLtMatmulDescAttributes_t matmulDescA = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;
  hipblasLtMatmulDescAttributes_t matmulDescB = HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;
#if defined(HIPBLASLT_VEC_EXT) && ROCM_VERSION < 70000
  if (use_rowwise) {
    matmulDescA = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT;
    matmulDescB = HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER_VEC_EXT;
  }
#else
  // rowwise isn't supported using cublaslt or older hipblaslt
  TORCH_INTERNAL_ASSERT(use_rowwise == false, "rowwise scaled_gemm not supported with blaslt");
#endif
  computeDesc.setAttribute(matmulDescA, mat1_scale_ptr);
  computeDesc.setAttribute(matmulDescB, mat2_scale_ptr);
  if (result_scale_ptr != nullptr) {
    computeDesc.setAttribute(HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, result_scale_ptr);
  }
  HipBlasLtMatrixLayout Adesc(ScalarTypeToCudaDataType(mat1_dtype), m, k, mat1_ld, transa == 't');
  HipBlasLtMatrixLayout Bdesc(ScalarTypeToCudaDataType(mat2_dtype), k, n, mat2_ld, transb == 't');
  // Cdesc is unused, beta is 0. But hipblaslt needs this set to something reasonable.
  HipBlasLtMatrixLayout Cdesc(ScalarTypeToCudaDataType(result_dtype), m, n, result_ld);
  HipBlasLtMatrixLayout Ddesc(ScalarTypeToCudaDataType(result_dtype), m, n, result_ld);
  if (bias_ptr) {
    computeDesc.setAttribute(HIPBLASLT_MATMUL_DESC_BIAS_POINTER, bias_ptr);
    computeDesc.setAttribute(HIPBLASLT_MATMUL_DESC_EPILOGUE, HIPBLASLT_EPILOGUE_BIAS);
    computeDesc.setAttribute(HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, ScalarTypeToCudaDataType(bias_dtype));
  }

#ifdef HIPBLASLT_HAS_ORDER_COL16
  if (mat1_is_swizzled) {
    Adesc.setAttribute(HIPBLASLT_MATRIX_LAYOUT_ORDER, HIPBLASLT_ORDER_COL16_4R16);
  }
  if (mat2_is_swizzled) {
    Bdesc.setAttribute(HIPBLASLT_MATRIX_LAYOUT_ORDER, HIPBLASLT_ORDER_COL16_4R16);
  }
#endif

  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA();
  size_t workspaceSize = _getWorkspaceSize();
  auto& allocator = *::c10::hip::HIPCachingAllocatorMasqueradingAsCUDA::get();
  auto workspace = allocator.allocate(workspaceSize);
  auto workspace_ptr = workspace.mutable_get();
  TORCH_CHECK(workspace_ptr != nullptr, "OOM trying to allocate workspace for cublaslt");

  HipBlasLtMatmulPreference preference;
  preference.setAttribute(HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);
  hipblasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  hipblasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
  TORCH_CUDABLAS_CHECK(hipblasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Ddesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  if (returnedResult == 0) {
    // hipblaslt might be able to recover by returning all algos
    std::vector<hipblasLtMatmulHeuristicResult_t> all_algos;
    TORCH_CUDABLAS_CHECK(hipblaslt_ext::getAllAlgos(
        ltHandle,
        hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        _cublasOpFromChar(transa),
        _cublasOpFromChar(transb),
        ScalarTypeToCudaDataType(mat1_dtype),
        ScalarTypeToCudaDataType(mat2_dtype),
        // C is nullptr and beta=0, so set to something reasonable. See above.
        //ScalarTypeToCudaDataType(bias_dtype),
        ScalarTypeToCudaDataType(result_dtype),
        ScalarTypeToCudaDataType(result_dtype),
        HIPBLAS_COMPUTE_32F,
        all_algos));
    if (all_algos.size() == 0) {
      TORCH_CUDABLAS_CHECK(HIPBLAS_STATUS_NOT_SUPPORTED);
    }
    // pick first valid solution
    bool found = false;
    for (size_t i = 0; i < all_algos.size(); i++) {
        size_t ret_workspace_size = 0;
        auto is_valid_status = hipblaslt_ext::matmulIsAlgoSupported(
                ltHandle,
                computeDesc.descriptor(),
                &alpha_val,
                Adesc.descriptor(),
                Bdesc.descriptor(),
                &beta_val,
                Cdesc.descriptor(),
                Ddesc.descriptor(),
                all_algos[i].algo,
                ret_workspace_size);
        if (is_valid_status == HIPBLAS_STATUS_SUCCESS) {
            if (ret_workspace_size <= workspaceSize) {
                heuristicResult = all_algos[i];
                found = true;
                break;
            }
        }
    }
    TORCH_CHECK(found, "could not find valid hipblaslt solution");
  }
  hipblasStatus_t cublasStatus = hipblasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      &alpha_val,
      mat1_ptr,
      Adesc.descriptor(),
      mat2_ptr,
      Bdesc.descriptor(),
      &beta_val,
      result_ptr, // unused, since beta_val is 0, but hipblaslt can't handle nullptr
      Cdesc.descriptor(),
      result_ptr,
      Ddesc.descriptor(),
      &heuristicResult.algo,
      workspace_ptr,
      workspaceSize,
      stream);
  TORCH_CHECK(
      cublasStatus == HIPBLAS_STATUS_SUCCESS,
      "CUDA error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling hipblasLtMatmul with transpose_mat1 ",
      transa,
      " transpose_mat2 ",
      transb,
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " mat1_ld ",
      mat1_ld,
      " mat2_ld ",
      mat2_ld,
      " result_ld ",
      result_ld,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
  return;
}

Tensor&
_scaled_mm_out(const Tensor& mat1, const Tensor& mat2,
          bool mat1_is_swizzled,
          bool mat2_is_swizzled,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          Tensor& out) {
  // Check sizes
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  // Check what type of scaling we are doing based on inputs
  ScalingType scaling_choice = get_scaling_type(scale_a, scale_b, mat1.size(0), mat2.size(1));
  TORCH_INTERNAL_ASSERT(scaling_choice != ScalingType::Error, "Scaling type not supported");

  TORCH_CHECK(!scale_result || (scale_result->numel() == 1 && scale_result->scalar_type() == kFloat),
       "scale_result must be a float scalar");
  TORCH_CHECK(!bias || bias->numel() == mat2.sizes()[1], "Bias must be size ", mat2.sizes()[1],
       " but got ", bias->numel());
  TORCH_CHECK(
      mat1.sizes()[1] % 16 == 0,
      "Expected trailing dimension of mat1 to be divisible by 16 ",
      "but got mat1 shape: (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      ").");
  TORCH_CHECK(mat2.sizes()[0] % 16 == 0 && mat2.sizes()[1] % 16 == 0, "mat2 shape (", mat2.sizes()[0], "x",
       mat2.sizes()[1], ") must be divisible by 16");
  // Check types
  TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  TORCH_CHECK(isFloat8Type(mat1.scalar_type()), "Expected mat1 to be Float8 matrix got ", mat1.scalar_type());
  TORCH_CHECK(isFloat8Type(mat2.scalar_type()), "Expected mat2 to be Float8 matrix got ", mat2.scalar_type());
  if (bias) {
    TORCH_CHECK(out.scalar_type() != kFloat, "Bias is not supported when out_dtype is set to Float32");
    TORCH_CHECK(bias->scalar_type() == ScalarType::BFloat16 || bias->scalar_type() == ScalarType::Half,
         "Bias must be either Half or BFloat16, but got ", bias->scalar_type());
    TORCH_CHECK((out.scalar_type() != kFloat && out.scalar_type() != ScalarType::BFloat16) ||
          bias->scalar_type() == ScalarType::BFloat16,
          "Bias must be BFloat16 to compute ", out.scalar_type(), " output, but got ", bias->scalar_type());
    TORCH_CHECK(out.scalar_type() != ScalarType::Half || bias->scalar_type() == ScalarType::Half,
          "Bias must be Float16 to compute ", out.scalar_type(), " output, but got ", bias->scalar_type());
  }
  {
    auto bias_ = bias.value_or(Tensor());
    auto scale_result_ = scale_result.value_or(Tensor());

    // NOLINTNEXTLINE(*c-array*)
    TensorArg targs[]{{out, "out", 0}, {mat1, "mat1", 1}, {mat2, "mat2", 2},
                      {bias_, "bias", 3}, {scale_a, "scale_a", 4}, {scale_b, "scale_b", 5},
                      {scale_result_, "scale_result", 6}};
    checkAllSameGPU(__func__, targs);
  }
  // Validation checks have passed lets resize the output to actual size
  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

  // If any of M, K, N is 0 - return early (the tensorwise/rowwise float8 gemm kernels
  // do not support this case).
  if (mat1_sizes[0] == 0 || mat1_sizes[1] == 0 || mat2_sizes[1] == 0) {
    // `out` was created with `at::empty`. In the case where we are multiplying
    // MxK by KxN and K is the zero dim, we need to initialize here to properly
    // return a tensor of zeros.
    if (mat1_sizes[1] == 0) {
      out.zero_();
    }

    return out;
  }

  if (scaling_choice == ScalingType::RowWise) {
    // For ROCm, match behavior of f8f8bf16_rowwise type checking, for unit test purposes.
    Tensor b = mat2;
    if (_scaled_mm_is_fnuz()) {
      TORCH_CHECK(b.dtype() == at::kFloat8_e4m3fnuz);
    }
    else {
      TORCH_CHECK(b.dtype() == at::kFloat8_e4m3fn);
    }
    // Until more than bf16 is supported.
    TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16,
         "hipblaslt rowwise _scaled_mm only supports BFloat16 output but got ", out.scalar_type());
  }

  cublasCommonArgs args(mat1, mat2, mat1_is_swizzled, mat2_is_swizzled, out, scale_a, scale_b, scale_result);
  const auto out_dtype_ = args.result->scalar_type();
  TORCH_CHECK(args.transa == 't' && args.transb == 'n', "Only multiplication of row-major and column-major matrices is supported by cuBLASLt");

  {
    _scaled_gemm(
        args.transa,
        args.transb,
        args.m,
        args.n,
        args.k,
        args.mata->data_ptr(),
        args.scale_mata_ptr,
        args.lda,
        args.mata->scalar_type(),
        args.scale_mata_dtype.value(),
        args.mata_is_swizzled,
        args.matb->data_ptr(),
        args.scale_matb_ptr,
        args.ldb,
        args.matb->scalar_type(),
        args.scale_matb_dtype.value(),
        args.matb_is_swizzled,
        bias ? bias->data_ptr(): nullptr,
        bias ? bias->scalar_type() : isFloat8Type(out_dtype_) ? at::ScalarType::Half : out_dtype_,
        args.result->data_ptr(),
        args.scale_result_ptr,
        args.result_ld,
        out_dtype_,
        scaling_choice == ScalingType::RowWise);
  }

  return out;
}

Tensor
swizzle_scaled_mm(const Tensor& mat_a, const Tensor& mat_b,
          bool mat1_is_swizzled,
          bool mat2_is_swizzled,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));
  return _scaled_mm_out(mat_a, mat_b, mat1_is_swizzled, mat2_is_swizzled, scale_a, scale_b, bias, scale_result, out_dtype, out);
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::swizzle_mm", &swizzle_mm);
  m.impl("torchao::swizzle_scaled_mm", &swizzle_scaled_mm);
}
#endif // USE_ROCM
