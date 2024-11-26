from pathlib import Path

import modal


app = modal.App("torchao-sam-2-cli")


REPO_ROOT = Path(__file__).parent
TARGET = "/root/"

N_H100 = 8

COMMIT_SHA = "cbc099dd73291fbd51f08b7b6f9360420f511890"
SCRIPT_URL = f"https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/{COMMIT_SHA}/train_gpt2.py"

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
    },
    #     TARGET + "logs": logs,
    #     # mount the caches of torch.compile and friends
    #     "/root/.nv": modal.Volume.from_name("nanogpt-nv-cache", create_if_missing=True),
    #     "/root/.triton": modal.Volume.from_name(
    #         "nanogpt-triton-cache", create_if_missing=True
    #     ),
    #     "/root/.inductor-cache": modal.Volume.from_name(
    #         "nanogpt-inductor-cache", create_if_missing=True
    #     ),
    # },
    timeout=30 * 60,
)
def train(input_bytes):
    import torch
    import torchao
    import os

    # module_path = torchao.__file__
    # print(f"The path to the torchao module is: {module_path}")
    # for root, dirs, files in os.walk(os.path.dirname(module_path)):
    #     for file in files:
    #         print(os.path.join(root, file))

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
    # List all files and directories in the current directory
    current_directory_contents = current_directory.iterdir()
    # Print the contents
    for item in current_directory_contents:
        print(item)

    from torchao import _models
    print(dir(_models))
    from torchao._models import sam2
    print(dir(sam2))
    # from torchao._models.sam2 import configs
    # print(dir(configs))
    print("torch.__version__")
    print(torch.__version__)

    with open('/tmp/dog.jpg', 'wb') as file:
        file.write(input_bytes)

    import sys
    sys.path.append(".")
    from cli import main as cli_main
    cli_main(Path(TARGET) / Path("checkpoints"),
             model_type="large",
             input_path="/tmp/dog.jpg",
             output_path="/tmp/dog_masked_2.png",
             verbose=True)
          
    return bytearray(open('/tmp/dog_masked_2.png', 'rb').read())

@app.local_entrypoint()
def main(input_path, output_path):
    bytes = train.remote(open(input_path, 'rb').read())
    with open(output_path, "wb") as file:
        file.write(bytes)
