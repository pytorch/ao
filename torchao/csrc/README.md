# Custom C++/CUDA Extensions

This folder is an example of how to integrate your own custom kernels into ao such that
1. They work on as many devices and operating systems as possible
2. They compose with `torch.compile()` without graph breaks

The goal is that you can focus on just writing your custom CUDA or C++ kernel and we can package it up so it's available via `torchao.ops.your_custom_kernel`.

To learn more about custom ops in PyTorch you can refer to the [PyTorch Custom Operators Landing Page](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)

## How to add your own kernel in ao

We've integrated several kernels which you can use as a template for your own kernels. `tensor_core_tiled_layout` is the most straight-forward to get started with.

1. Install the cudatoolkit https://anaconda.org/conda-forge/cudatoolkit
2. In `csrc/cuda` author your custom kernel and ensure you expose a `TORCH_LIBRARY_IMPL` which will expose `torchao::your_custom_kernel`
3. In `torchao/ops.py`, define your op signature at the top of the file. You can refer to [this](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md) on how to write the signature correctly
4. `torchao/ops.py` is also where you'll expose the python API which your new end users will leverage
5. Write a new test in `test/test_ops.py` which most importantly needs to pass `opcheck()`, this ensures that your custom kernel composes out of the box with `torch.compile()`

And that's it! Once CI passes and your code merged you'll be able to point people to `torchao.ops.your_custom_kernel`. If you're working on an interesting kernel and would like someone else to handle the release and package management please feel free to open an issue.

If you'd like to learn more please check out [torch.library](https://pytorch.org/docs/main/library.html)

Note: All CUTLASS-based kernels should have `cutlass` in the name of their `.cu` files e.g. `rowwise_scaled_linear_cutlass_s4s4.cu`

## Required dependencies

The important dependencies are already taken care of in our CI so feel free to test in CI directly

1. cudatoolkit so you can build your own custom extensions locally. We highly recommend using https://anaconda.org/conda-forge/cudatoolkit for installation
2. manylinux with CUDA support. In your own Github actions you can integrate this support using `uses: pytorch/test-infra/.github/workflows/linux_job.yml@main`
