# Dynamic Dispatch for x86 CPU Kernels

## 1. Overview

This system enables runtime dispatch to ISA-optimized kernel implementations while maintaining a single codebase. Kernels are compiled for multiple ISA levels and selected at runtime based on CPU capabilities detected during build time.

## 2. ISA Levels

The dispatcher supports three ISA levels, defined in `setup.py`:

### 2.1. DEFAULT (Base Level)
- Portable fallback using scalar operations
- No specific ISA requirements
- Always compiled and available
- Fallback when higher ISA kernels cannot be used

### 2.2. AVX512 (Intermediate Level)
- **Requirements**: AVX-512F, AVX-512BW, AVX-512VL, AVX-512DQ, AVX-512-VNNI
- **Additional**: AMX tile accelerator (int8, tile, bf16 extensions)
- **Compiler**: GCC >= 11.2 with `-mavx512*` and `-mamx-*` support
- **Target**: Sapphire Rapids and later x86-64 CPUs
- **ISA probe**: Verifies `_mm512_dpbusd_epi32` compilation (VNNI intrinsic)

Notes:
- For non-GEMM kernels, this level does not require AVX512-VNNI or AMX.
- For GEMM kernels (DA8W4 linear, Float8 linear, etc.), this level also requires AVX512-VNNI and AMX. We do not treat AVX512-VNNI and AMX as separate ISA levels in the dynamic dispatch mechanism here because:
  - The GEMM kernels require AMX but we do not write AMX code explicitly. The kernels calls oneDNN's micro-kernel API (brgemm) from torch.
  - You only run into the VNNI code when AMX is available since it serves as a supplement to the primary AMX (brgemm) implementation. There is no VNNI-only kernel.

### 2.3. AVX10.2 (Highest Level)
- **Includes**: All AVX512 features + AVX10.2 enhancements
- **Key Feature**: Native FP8↔FP16 hardware conversion (`_mm256_cvthf8_ph`)
- **Compiler**: GCC >= 15 with `-mavx10.2` support
- **Target**: Diamond Rapids and later x86-64 CPUs
- **ISA probe**: Verifies `_mm256_cvthf8_ph` compilation (AVX10.2 intrinsic)

## 3. How to Add a New Kernel

### 3.1. Implement your kernel
- Create `torchao/csrc/cpu/aten_kernels/<kernel_name>_krnl.cpp`
- Use `#if defined(CPU_CAPABILITY_*)` guards to conditionally compile ISA-specific intrinsics

### 3.2. Update dispatch.cpp

**Define function pointer type:**
```cpp
using my_kernel_fn = return_type(*)(arg1_type, arg2_type, ...);
```

**Add to KernelDispatcher struct:**
```cpp
struct KernelDispatcher {
  // ... existing fields ...
  my_kernel_fn my_kernel;  // Add new field
};
```

**Declare kernel in all namespaces:**
Use the `declare_all_kernels(namespace_name)` macro to declare `declare_my_kernel_impl` in `DEFAULT`, `AVX512`, and `AVX10_2` namespaces.

**Add to get_kernel_dispatcher():**
Initialize the dispatcher table by populating `my_kernel` for each ISA level:
```cpp
KernelDispatcher& get_kernel_dispatcher() {
  static KernelDispatcher dispatcher = []() {
    // ... existing dispatch logic ...
#if defined(BUILD_AVX10_2)
    if (kHasAVX10_2) {
      d = {/* ... existing ... */, AVX10_2::my_kernel_impl};
      return d;
    }
#endif
#if defined(BUILD_AVX512)
    if (kHasAVX512) {
      d = {/* ... existing ... */, AVX512::my_kernel_impl};
      return d;
    }
#endif
    d = {/* ... existing ... */, DEFAULT::my_kernel_impl};
    return d;
  }();
  return dispatcher;
}
```

**Create dispatch wrapper:**
```cpp
declare_my_kernel_impl {
  return get_kernel_dispatcher().my_kernel(arg1, arg2, ...);
}
```

**Register with PyTorch:**
```cpp
TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::my_kernel", &my_kernel_impl);
}
```

### 3.3. Build Integration

No additional setup needed in `setup.py`. The build system automatically:
- Detects compiler ISA support via probe snippets
- Compiles kernel files with appropriate `-mavx512*` and `-mavx10.2` flags
- Links ISA-specific `.o` files as extra objects to the main extension

## Build System Design

### ISA Detection (setup.py: X86KernelBuild)

**Compiler Discovery:**
1. Check `$CXX` environment variable
2. Fall back to `g++` on `$PATH`
3. Verify compiler is functional

**ISA Capability Probing:**
- Compile small C++ snippets that exercise ISA intrinsics
- **AVX512 probe**: Attempts `_mm512_dpbusd_epi32` with `-mavx512*` flags
- **AVX10.2 probe**: Attempts `_mm256_cvthf8_ph` with `-mavx10.2` flags
- Selects highest supported ISA level for compilation

**ISA-specific Object Compilation:**
- For each ISA level the compiler supports:
  - Copy each `*_krnl.cpp` to `build_temp/cpu_isa_<LEVEL>/`
  - Compile with `-DCPU_CAPABILITY_<LEVEL>` and appropriate `-m...` flags
  - Link resulting `.o` files as extra objects to `torchao._C` extension
- Compilation is parallelized via `ThreadPoolExecutor`

### Compilation Flags

**Explicit ISA flags replace `-march=` codenames:**
- AVX512: `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mamx-int8 -mamx-tile -mamx-bf16`
- AVX10.2: (AVX512 flags) + `-mavx10.2`

**Compile-time defines:**
- `-DBUILD_AVX512`: Set when compiler supports AVX512
- `-DBUILD_AVX10_2`: Set when compiler supports AVX10.2
- `-DCPU_CAPABILITY_*`: Set during each ISA-specific object compilation

### Runtime Dispatch (dispatch.cpp)

**Dispatcher initialization:**
- Static initialization of `KernelDispatcher` on first call to `get_kernel_dispatcher()`
- Checks runtime capabilities (`kHasAVX10_2`, `kHasAVX512` from `utils.h`)
- Selects highest available ISA level
- Stores function pointers in dispatcher table

**Dispatch overhead:**
- First call: One-time initialization of dispatcher table
- Subsequent calls: Single indirect function call (~1-2 CPU cycles)
- Negligible compared to kernel execution time

## 4. File Organization

```
torchao/csrc/cpu/aten_kernels/
  ├── dispatch.cpp           # Dynamic dispatch logic and wrappers
  ├── dynamic_dispatch.md    # This file
  ├── utils.h                # kHasAVX10_2, kHasAVX512, CPU intrinsics
  ├── <kernel>_krnl.cpp      # Kernel implementations (3 ISA variants each)
  └── ...
```

## 5. Manual Dispatch for Testing

Set the `TORCHAO_CPU_DISPATCH` environment variable to override the default ISA selection. This enables testing all code paths on high-capability platforms.

**Available keys:** `DEFAULT`, `AVX512`, `AMX`, `AVX10_2`, `AUTO`

**Behavior:**
- `DEFAULT`: Force scalar fallback implementation
- `AVX512`: Use AVX512 for non-GEMM kernels; disable AMX for GEMM kernels (scalar fallback)
- `AMX`: Use AVX512 for non-GEMM kernels; allow AMX for GEMM kernels (subject to hardware availability)
- `AVX10_2`: Use AVX10.2 implementation with hardware Float8 features.
- `AUTO`: Default behavior (automatic selection based on CPU capabilities)

Set `TORCHAO_CPU_DISPATCH_DEBUG=1` to print the selected ISA level during initialization.
