from pathlib import Path
import json
import fire

import modal

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

app = modal.App("torchao-sam-2-cli", image=image)

checkpoints = modal.Volume.from_name("torchao-sam-2-cli-checkpoints", create_if_missing=True)
data = modal.Volume.from_name("torchao-sam-2-cli-data", create_if_missing=True)


@app.cls(
    gpu="H100",
    container_idle_timeout=20 * 60,
    timeout=20 * 60,
    volumes={
        TARGET + "checkpoints": checkpoints,
        TARGET + "data": data,
    },
)
class Model:
    model_type: str = modal.parameter(default="large")
    points_per_batch: int = modal.parameter(default=1024)
    fast: int = modal.parameter(default=0)
    furious: int = modal.parameter(default=0)

    def calculate_file_hash(self, file_path, hash_algorithm='sha256'):
        import hashlib
        """Calculate the hash of a file."""
        hash_func = hashlib.new(hash_algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def download_file(self, url, filename):
        import subprocess
        command = f"wget -O {filename} {url}"
        subprocess.run(command, shell=True, check=True)

    @modal.build()
    @modal.enter()
    def build(self):
        import os
        from torchao._models.sam2.build_sam import build_sam2
        from torchao._models.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        download_url_branch = "climodal2"
        download_url = f"{DOWNLOAD_URL_BASE}/{download_url_branch}/"
        download_url += "examples/sam2_amg_server/"

        h = self.calculate_file_hash(TARGET + "data/cli.py")
        print("cli.py hash: ", h)
        if h != "b38d60cb6fad555ad3c33081672ae981a5e4e744199355dfd24d395d20dfefda":
            self.download_file(download_url + "cli.py", TARGET + "data/cli.py")

        h = self.calculate_file_hash(TARGET + "data/server.py")
        print("server.py hash: ", h)
        if h != "af33fdb9bcfe668b7764cb9c86f5fa9a799c999306e7c7e5b28c988b2616a0ae":
            self.download_file(download_url + "server.py", TARGET + "data/server.py")

        os.chdir(Path(TARGET + "data"))
        import sys
        sys.path.append(".")

        from server import model_type_to_paths
        from server import set_fast
        from server import set_furious


        device = "cuda"
        checkpoint_path = Path(TARGET) / Path("checkpoints")
        sam2_checkpoint, model_cfg = model_type_to_paths(checkpoint_path, self.model_type)
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_batch=self.points_per_batch, output_mode="uncompressed_rle")
        self.mask_generator = mask_generator
        if self.fast:
            set_fast(mask_generator)
        if self.furious:
            set_furious(mask_generator)

    @modal.method()
    def inference_rle(self, input_bytes) -> dict:
        import os
        os.chdir(Path(TARGET + "data"))
        import sys
        sys.path.append(".")
        from server import file_bytes_to_image_tensor
        from server import masks_to_rle_dict
        image_tensor = file_bytes_to_image_tensor(input_bytes)
        masks = self.mask_generator.generate(image_tensor)
        return masks_to_rle_dict(masks)

    @modal.method()
    def inference(self, input_bytes, output_format='png'):
        import os
        os.chdir(Path(TARGET + "data"))
        import sys
        sys.path.append(".")
        from server import file_bytes_to_image_tensor
        from server import show_anns
        image_tensor = file_bytes_to_image_tensor(input_bytes)
        masks = self.mask_generator.generate(image_tensor)

        import matplotlib.pyplot as plt
        from io import BytesIO
        from torchao._models.sam2.utils.amg import rle_to_mask
        plt.figure(figsize=(image_tensor.shape[1]/100., image_tensor.shape[0]/100.), dpi=100)
        plt.imshow(image_tensor)
        show_anns(masks, rle_to_mask)
        plt.axis('off')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format=output_format)
        buf.seek(0)
        return buf.getvalue()


def main(input_path, output_path, fast=False, furious=False, model_type="large", output_rle=False):
    input_bytes = bytearray(open(input_path, 'rb').read())
    try:
        model = modal.Cls.lookup("torchao-sam-2-cli", "Model")()
    except modal.exception.NotFoundError:
        print("Can't find running app. To deploy the app run the following command. Note that this costs money! See https://modal.com/pricing")
        print("modal deploy cli_on_modal.py")
        return

    if output_rle:
        output_dict = model.inference_rle.remote(input_bytes)
        with open(output_path, "w") as file:
            file.write(json.dumps(output_dict, indent=4))
    else:
        output_bytes = model.inference.remote(input_bytes)
        with open(output_path, "wb") as file:
            file.write(output_bytes)


if __name__ == "__main__":
    fire.Fire(main)
