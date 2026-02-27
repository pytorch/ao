from torchao.experimental.ops.mps.utils import _load_torchao_mps_lib

_load_torchao_mps_lib()

# Import to register Meta implementations
from torchao.experimental.ops.mps import mps_op_lib  # noqa: F401
