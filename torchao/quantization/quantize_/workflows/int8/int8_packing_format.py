from enum import Enum


class Int8PackingFormat(str, Enum):
    """Packing format for quantized data in Int8 Tensor subclasses in torchao, represents how
    the values in quantized data are packed and laid out in memory.
    """

    """
    csr_sparse is referring to the format used by csr kernels
    """
    CSR_SPARSE = "csr_sparse"
