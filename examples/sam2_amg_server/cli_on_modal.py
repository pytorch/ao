from pathlib import Path

import modal

app = modal.App("torchao-sam-2-cli")

TARGET = "/root/"
DOWNLOAD_URL_BASE = "https://raw.githubusercontent.com/pytorch/ao/refs/heads"

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

checkpoints = modal.Volume.from_name("torchao-sam-2-cli-checkpoints", create_if_missing=True)
data = modal.Volume.from_name("torchao-sam-2-cli-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={
        TARGET + "checkpoints": checkpoints,
        TARGET + "data": data,
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
def inference(input_bytes, fast, furious, model_type):
    import os
    import subprocess
    import hashlib

    def calculate_file_hash(file_path, hash_algorithm='sha256'):
        """Calculate the hash of a file."""
        hash_func = hashlib.new(hash_algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def download_file(url, filename):
        command = f"wget -O {filename} {url}"
        subprocess.run(command, shell=True, check=True)

    download_url_branch = "climodal2"
    download_url = f"{DOWNLOAD_URL_BASE}/{download_url_branch}/"
    download_url += "examples/sam2_amg_server/"

    h = calculate_file_hash(TARGET + "data/cli.py")
    print("cli.py hash: ", h)
    if h != "b38d60cb6fad555ad3c33081672ae981a5e4e744199355dfd24d395d20dfefda":
        download_file(download_url + "cli.py", TARGET + "data/cli.py")

    h = calculate_file_hash(TARGET + "data/server.py")
    print("server.py hash: ", h)
    if h != "af33fdb9bcfe668b7764cb9c86f5fa9a799c999306e7c7e5b28c988b2616a0ae":
        download_file(download_url + "server.py", TARGET + "data/server.py")

    os.chdir(Path(TARGET + "data"))

    import sys
    sys.path.append(".")
    from cli import main_headless
    output_bytes = main_headless(Path(TARGET) / Path("checkpoints"),
                                 model_type=model_type,
                                 input_bytes=input_bytes,
                                 verbose=True,
                                 fast=fast,
                                 furious=furious)
    print("Returning image as bytes.")
    return output_bytes


@app.local_entrypoint()
def main(input_path, output_path, fast=False, furious=False, model_type="large"):
    output_bytes = inference.remote(bytearray(open(input_path, 'rb').read()), fast, furious, model_type)
    with open(output_path, "wb") as file:
        file.write(output_bytes)
