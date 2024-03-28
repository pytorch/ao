from setuptools import setup

REQUIREMENTS = [
    "torch",
    "triton",
    "numpy",
]
setup(
    name="galore-fused",
    version="1.0",
    description="Fused kernels for GaLore Optimizers",
    url="https://github.com/pytorch-labs/ao/prototype/galore",
    author="Jerome Ku",
    author_email="jerome.ku@gmail.com",
    license="Apache 2.0",
    packages=["galore_fused"],
    install_requires=REQUIREMENTS,
)