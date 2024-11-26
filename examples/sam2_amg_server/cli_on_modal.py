from pathlib import Path

import modal

app = modal.App("torchao-sam-2-cli")

TARGET = "/root/"

image = (
    modal.Image.debian_slim(python_version="3.12.7")
    .pip_install("numpy<3", "tqdm")
    .pip_install(
        "torch",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu124",  # tested with torch-2.6.0.dev20241120
    )
    .pip_install(
        "torchvision",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu124",  # tested with torch-2.6.0.dev20241120
    )
    .apt_install("git")
    .apt_install("libopencv-dev")
    .apt_install("python3-opencv")
    .run_commands(["git clone https://github.com/pytorch/ao.git /tmp/ao_src"])
    .run_commands(["cd /tmp/ao_src; python setup.py develop"])
    .pip_install(
        "gitpython",
    )
    .apt_install("wget")
    .run_commands([f"wget https://raw.githubusercontent.com/pytorch/ao/refs/heads/main/examples/sam2_amg_server/requirements.txt"])
    .pip_install_from_requirements(
        'requirements.txt',
    )
)

checkpoints = modal.Volume.from_name("checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",
    volumes={
        TARGET + "checkpoints": checkpoints,
        # # mount the caches of torch.compile and friends
        # "/root/.nv": modal.Volume.from_name("torchao-sam-2-cli-nv-cache", create_if_missing=True),
        # "/root/.triton": modal.Volume.from_name(
        #     "torchao-sam-2-cli-triton-cache", create_if_missing=True
        # ),
        # "/root/.inductor-cache": modal.Volume.from_name(
        #     "torchao-sam-2-cli-inductor-cache", create_if_missing=True
        # ),
    },
    timeout=60 * 60,
)
def eval(input_bytes, fast, furious):
    import torch
    import torchao
    import os

    import subprocess
    from pathlib import Path
    from git import Repo

    def download_file(url, filename):
        command = f"wget -O {filename} {url}"
        subprocess.run(command, shell=True, check=True)

    os.chdir(Path(TARGET))
    download_file("https://raw.githubusercontent.com/pytorch/ao/refs/heads/climodal1/examples/sam2_amg_server/cli.py", "cli.py")
    download_file("https://raw.githubusercontent.com/pytorch/ao/refs/heads/climodal1/examples/sam2_amg_server/server.py", "server.py")
    # Create a Path object for the current directory
    current_directory = Path('.')

    with open('/tmp/dog.jpg', 'wb') as file:
        file.write(input_bytes)

    import sys
    sys.path.append(".")
    from cli import main as cli_main
    cli_main(Path(TARGET) / Path("checkpoints"),
             model_type="large",
             input_path="/tmp/dog.jpg",
             output_path="/tmp/dog_masked_2.png",
             verbose=True,
             fast=fast,
             furious=furious)
          
    return bytearray(open('/tmp/dog_masked_2.png', 'rb').read())

@app.local_entrypoint()
def main(input_path, output_path, fast=False, furious=False):
    bytes = eval.remote(open(input_path, 'rb').read(), fast, furious)
    with open(output_path, "wb") as file:
        file.write(bytes)
