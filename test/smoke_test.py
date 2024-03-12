"""Run smoke tests"""

import sys
from pathlib import Path

import torch
import torchao

SCRIPT_DIR = Path(__file__).parent


def main() -> None:
    # print(f"torchvision: {torchvision.__version__}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")

    # # Turn 1.11.0aHASH into 1.11 (major.minor only)
    # version = ".".join(torchvision.__version__.split(".")[:2])
    # if version >= "0.16":
    #     print(f"{torch.ops.image._jpeg_version() = }")
    #     assert torch.ops.image._is_compiled_against_turbo()

    # smoke_test_torchvision()
    # smoke_test_torchvision_read_decode()
    # smoke_test_torchvision_resnet50_classify()
    # smoke_test_torchvision_decode_jpeg()
    # if torch.cuda.is_available():
    #     smoke_test_torchvision_decode_jpeg("cuda")
    #     smoke_test_torchvision_resnet50_classify("cuda")

    #     # TODO: remove once pytorch/pytorch#110436 is resolved
    #     if sys.version_info < (3, 12, 0):
    #         smoke_test_compile()

    # if torch.backends.mps.is_available():
    #     smoke_test_torchvision_resnet50_classify("mps")


if __name__ == "__main__":
    main()
