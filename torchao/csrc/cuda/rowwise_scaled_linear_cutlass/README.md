This directory is intended to contain implementations for all of the
CUTLASS-based row-wise scaled linear operators, for non-sparse inputs
of both same and mixed data types.

The implementation is through single kernel per SM generation, that
should reside in `rowwise_scaled_linear_kernel_cutlass.cuh` file.  At
the moment, only SM8.x architectures are supported, through
`rowwise_scaled_linear_kernel_cutlass_sm8x` kernel, but the SM9.x, and
eventually higher, can and will be supported too.

The rest of source files, besides
`rowwise_scaled_linear_kernel_cutlass.cuh` file, contain just the
corresponding template instantiation and PyTorch operator declaration
for given operator.

In order to support new combination of data types, copy one of
existing `.cu` files, for example
`rowwise_scaled_linear_kernel_cutlass_s8s4.cu`, rename the new file,
as well as operator to be defined inside, to reflect data types to be
supported, and also change `using ElementA` and `using ElementB`
directives accordingly.

In the `.cuh` file, looking from the bottom up, the changes needed as
follows:

1. Optionally, in the `rowwise_scaled_linear_cutlass_check_inputs`
template, changes may be needed at the places where the last dimension
of first operand is checked - but this check will have to be updated
only for inputs of mixed data types, where wider data type is not
exactly two times wider than the other data type.
2. In the `select_config` template, a section should be added to
choose optimal configuration(s) for your kernel.  The configuration
selection is critical for performance of any CUTLASS-based kernel, so
this is where the most time should and will be spent when making
changes.
3. Optionally, in the `rowwise_scaled_linear_kernel_cutlass_sm8x`
template, `using Operator` directive may need to be adjusted; namely,
for some combination of operands, `OpMultiplyAdd` may have to be used.

After making these changes, the test file
`tests/test_rowwise_scaled_linear_cutlass.py` should be changed too -
add a test for the new operator alike to existing tests.

To restrict build times, the implementation in `.cuh` file has some
restrictions at the moment, for example: scale tensors could be only
of `float16` or `bfloat16` data types, the output is produces to be of
the same data type as first input scale tensor, scale tensors are not
optional while bias is optional, etc.  If any of these restrictions
should be removed, or if any alike changes are needed, or if support
for other architectures is needed, or if you need any kind of help in
extending this code to support other data type combinations - get in
touch with the developers.
