from pathlib import Path
import time
import json
import fire

from fastapi import File, UploadFile
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
    .run_commands(["git clone https://github.com/pytorch/ao.git /tmp/ao_src_0"])
    .run_commands(["cd /tmp/ao_src_0; git checkout 1be4307db06d2d7e716d599c1091a388220a61e4"])
    .run_commands(["cd /tmp/ao_src_0; python setup.py develop"])
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
exported_models = modal.Volume.from_name("torchao-sam-2-exported-models", create_if_missing=True)
traces = modal.Volume.from_name("torchao-sam-2-traces", create_if_missing=True)


@app.cls(
    gpu="H100",
    container_idle_timeout=20 * 60,
    concurrency_limit=1,
    allow_concurrent_inputs=1,
    timeout=20 * 60,
    volumes={
        TARGET + "checkpoints": checkpoints,
        TARGET + "data": data,
        TARGET + "exported_models": exported_models,
        TARGET + "traces": traces,
    },
)
class Model:
    model_type: str = modal.parameter(default="large")
    points_per_batch: int = modal.parameter(default=1024)
    # fast: int = modal.parameter(default=0)
    # furious: int = modal.parameter(default=0)

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

    def download_and_verify_file(self,
                                 url,
                                 filename,
                                 hash_value,
                                 hash_algorithm='sha256'):
        if Path(filename).exists():
            h = self.calculate_file_hash(filename, hash_algorithm)
            if hash_value == h:
                return
        # Here either the file doesn't exist or the file
        # has the wrong hash, so we try to download it again.
        self.download_file(url, filename)
        h = self.calculate_file_hash(filename, hash_algorithm)
        if h != hash_value:
            raise ValueError(f"Url {url} doesn't contain file with "
                             f"{hash_algorithm} hash of value "
                             f"{hash_value}")

    @modal.build()
    @modal.enter()
    def build(self):
        import os
        from torchao._models.sam2.build_sam import build_sam2
        from torchao._models.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        download_url_branch = "main"
        download_url = f"{DOWNLOAD_URL_BASE}/{download_url_branch}/"
        download_url = download_url + "examples/sam2_amg_server"

        file_hashes = {
            'cli.py': "8bce88807fe360babd7694f7ee009d7ea6cdc150a4553c41409589ec557b4c4b",
            'server.py': "2d79458fabab391ef45cdc3ee9a1b62fea9e7e3b16e0782f522064d6c3c81a17",
            'compile_export_utils.py': "552c422a5c267e57d9800e5080f2067f25b4e6a3b871b2063a2840033f4988d0",
        }

        for f in file_hashes:
            self.download_and_verify_file(f"{download_url}/{f}",
                                          TARGET + f"data/{f}",
                                          file_hashes[f])

        os.chdir(Path(TARGET + "data"))
        import sys
        sys.path.append(".")

        from server import model_type_to_paths
        from compile_export_utils import set_furious

        device = "cuda"
        checkpoint_path = Path(TARGET) / Path("checkpoints")
        sam2_checkpoint, model_cfg = model_type_to_paths(checkpoint_path, self.model_type)
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_batch=self.points_per_batch, output_mode="uncompressed_rle")
        from compile_export_utils import load_exported_model
        mask_generator = load_exported_model(mask_generator,
                                             Path(TARGET) / Path("exported_models"),
                                             "amg",  # task_type
                                             furious=True,
                                             batch_size=1,
                                             points_per_batch=1024)
        self.mask_generator = mask_generator
        import os
        import torch
        import numpy as np
        import sys
        os.chdir(Path(TARGET + "data"))
        sys.path.append(".")
        from torchvision import io as tio
        from torchvision.transforms.v2 import functional as tio_F
        from server import masks_to_rle_dict
        from server import profiler_runner

        self.np = np
        self.tio = tio
        self.tio_F = tio_F
        self.torch = torch
        self.masks_to_rle_dict = masks_to_rle_dict
        self.profiler_runner = profiler_runner

    # @app.post("/upload_rle")
    # @modal.method()
    @modal.web_endpoint(docs=True, method="POST")
    async def upload_rle(self, image: UploadFile):
        from torch.autograd.profiler import record_function

        def upload_rle_inner(img_bytes):
            with record_function("asarray"):
                image_array = self.np.asarray(img_bytes, dtype=self.np.uint8)
            with record_function("from_numpy"):
                img_bytes_tensor = self.torch.from_numpy(image_array)

            with record_function("decode_jpeg"):
                image_tensor = self.tio.decode_jpeg(img_bytes_tensor,
                                                    device='cuda',
                                                    mode=self.tio.ImageReadMode.RGB)
            with record_function("to_dtype"):
                image_tensor = self.tio_F.to_dtype(image_tensor,
                                                   self.torch.float32,
                                                   scale=True)

            with record_function("generate"):
                masks = self.mask_generator.generate(image_tensor)
            with record_function("masks_to_rle_dict"):
                result = self.masks_to_rle_dict(masks)
            return result

        # return self.profiler_runner(TARGET + "traces/trace.json.gz", upload_rle_inner, bytearray(await image.read()))
        return upload_rle_inner(bytearray(await image.read()))

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


def main(input_paths, output_paths, output_rle=False):
    input_paths = open(input_paths).read().split("\n")
    output_paths = open(output_paths).read().split("\n")
    for input_path, output_path in zip(input_paths, output_paths):
        assert Path(input_path).exists()
        assert Path(output_path).exists()

    try:
        model = modal.Cls.lookup("torchao-sam-2-cli", "Model")()
    except modal.exception.NotFoundError:
        print("Can't find running app. To deploy the app run the following command. Note that this costs money! See https://modal.com/pricing")
        print("modal deploy cli_on_modal.py")
        return

    for input_path, output_path in zip(input_paths, output_paths):
        start = time.perf_counter()
        input_bytes = bytearray(open(input_path, 'rb').read())

        if output_rle:
            output_dict = model.inference_rle.remote(input_bytes)
            with open(output_path, "w") as file:
                # file.write(json.dumps(output_dict, indent=4))
                file.write(str(output_dict))
        else:
            output_bytes = model.inference.remote(input_bytes)
            with open(output_path, "wb") as file:
                file.write(output_bytes)
        end = time.perf_counter()
        print(end - start)


if __name__ == "__main__":
    fire.Fire(main)
