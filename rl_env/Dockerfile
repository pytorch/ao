# NOTE: PyTorch 2.10 and CUDA 13.0 are beyond this model's knowledge cutoff (Aug 2025).
# Verify the pytorch/pytorch image tag at https://hub.docker.com/r/pytorch/pytorch/tags
# and the wheel URL at https://download.pytorch.org/whl/ before building.

FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel

# Ensure Python 3.12.13 is the active interpreter.
# The pytorch/pytorch base image ships a conda-managed Python; verify with:
#   docker run --rm pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel python --version
# If the bundled version differs, uncomment the block below to install via deadsnakes:
#
# RUN apt-get update && apt-get install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && apt-get install -y python3.12 python3.12-dev python3.12-venv && \
#     update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
#     rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        # Required to build torchao CPU/CUDA extensions
        build-essential \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Full clone (no --depth) so git worktree can checkout arbitrary base commits
# for task environments. Pin to a tag/commit for reproducible builds, e.g.:
#   RUN git clone --branch v0.x.y https://github.com/pytorch/ao.git ao
RUN git clone --branch dvm-rl-env https://github.com/pytorch/ao.git ao

WORKDIR /workspace/ao

# Install torchvision (needed by several tests and benchmarks)
RUN pip install --no-cache-dir torchvision

# Install test/benchmark dependencies first (pandas, pytest, matplotlib, ninja, etc.)
# ninja and cmake here are Python wrappers; the system build-essential provides the
# actual compilers needed for the CUDA extension build below.
RUN pip install --no-cache-dir -r dev-requirements.txt

# Install torchao itself in editable mode so agent-modified source files are
# picked up by the interpreter without reinstalling.
# --no-build-isolation is required: CUDA extensions need torch at build time,
# which isn't available inside a PEP 517 isolated build environment.
RUN pip install --no-cache-dir -e . --no-build-isolation

# Install mini-swe-agent (provides the `mini` CLI used by run_task.py)
RUN pip install --no-cache-dir mini-swe-agent

# ANTHROPIC_API_KEY (and other model keys) must be passed at runtime:
#   docker run --gpus all -e ANTHROPIC_API_KEY=sk-ant-... torchao-bench
# Do NOT bake secrets into the image.