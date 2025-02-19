# Reproducing experiments locally

You can simply run `python reproduce_experiments.py <path/to/image_paths_file> <path/to/output_folder>`

`image_paths_file` needs to be a flat list of paths to images, for example

```
/home/$USER/data/sav_val/JPEGImages_24fps/sav_044979/00349.jpg
/home/$USER/data/sav_val/JPEGImages_24fps/sav_006751/00204.jpg
/home/$USER/data/sav_val/JPEGImages_24fps/sav_053118/00239.jpg
/home/$USER/data/sav_val/JPEGImages_24fps/sav_053391/00517.jpg
/home/$USER/data/sav_val/JPEGImages_24fps/sav_018487/00001.jpg
/home/$USER/data/sav_val/JPEGImages_24fps/sav_028552/00153.jpg
/home/$USER/data/sav_val/JPEGImages_24fps/sav_013729/00103.jpg
/home/$USER/data/sav_val/JPEGImages_24fps/sav_014662/00339.jpg
```

or whichever other files you'd like to use for study. For example you may consider the Segment Anything Video (SA-V) [Dataset](https://github.com/facebookresearch/sam2/tree/main/sav_dataset#download-the-dataset).

The experimental results will then be saved under `output_folder` in result.csv

# Reproducing experiments on Modal

For this you can run `modal_experiments.sh` after, but you'll want to experiments locally first to produce the meta annotations and exported ahead-of-time compiled binaries.

# Using the server locally
## Example curl command
```
curl -X POST http://127.0.0.1:5000/upload -F 'image=@/path/to/file.jpg' --output path/to/output.png
```

## Example script to collect rles

Start the server

```
python server.py ~/checkpoints/sam2 large --port <your_port> --host <your_hostname> --fast
```

Collect the rles

```
xargs -I {} curl -s -w "\n" -X POST http://<your_hostname>:<your_port>/upload_rle -F 'image=@{}' < image_paths > rle_masks
```

## mIoU scores on random subset of sav validation dataset

Experiments run on H100 and with batch size 1

| mode            | mIoU               | mask count mismatch | avg. ms per request | max. memory (MiB (%)) | batch size | points per batch |
| --------------  | -----------------  | ------------------- | ------------------- | --------------------- | ---------- | ---------------- |
|        baseline | 1.0                |   0                 | 863                 |  4013MiB (4%)         |  1         |   64             |
|              ao | 0.9999980926513672 |   6                 | 586                 |  3257MiB (3%)         |  1         |   64             |
|            fast | 0.993732988834381  | 191                 | 326                 | 27197MiB (27%)        |  1         | 1024             |
|            fast | 0.9937511086463928 | 194                 | 315                 | 27488MiB (28%)        | 16         | 1024             |
|  fast + furious | 0.9817246198654175 | 266                 | 120                 | 13616MiB (13%)        |  1         | 1024             |
|  fast + furious | 0.9794579744338989 | 274                 | 122                 | 13808MiB (14%)        | 16         | 1024             |

mask count mismatch counts the number of requests where the number of masks differ from the baseline.
For example, the baseline may have chosen to segment an image into 18 masks, but the fast variant produces 17 or 19.
We exclude these examples from the mIoU calculation.
Difference in mask count seem to stem from even only slight reorderings in compute. For example preprocessing on GPU instead of CPU.
A more relaxed way of measuring mIoU might be useful here to take into account slight differences in the number of masks, which may be caused by additional or missing sub-divisions.

The 'ao' mode is a copy of the baseline with modifications to make the code more compile-able and speed up run length encoding

### 0. Download checkpoints and install requirements

```
# From the top-level "ao" directory

# If necessary, create and activate a virtual environment
# Ex:
python -m venv venv && source venv/bin/activate

# Install requirements for this example
pip install -r examples/sam2_amg_server/requirements.txt

# If you have an older version of torch in your current environment, uninstall it first
pip uninstall torch

# Install torch nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Build ao from source for now
python setup.py develop

# On your mark, get set...
cd examples/sam2_amg_server/
```

Download `sam2.1_hiera_large.pt` from https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints and put it into `~/checkpoints/sam2`

### 1. Create a random subset of 1000 images
Using images with corresponding mask annotations, like from the Segment Anything Video (SA-V) [Dataset](https://github.com/facebookresearch/sam2/tree/main/sav_dataset#download-the-dataset) is suggested, to later compare any drop in accuracy using `--furious` (using `torch.float16`).
```
find sav_val -type f > sav_val_image_paths
shuf -n 1000 sav_val_image_paths > sav_val_image_paths_shuf_1000
```

### 2. Use the baseline (https://github.com/facebookresearch/sam2) to generate rles

Make sure you've installed https://github.com/facebookresearch/sam2

Start server
```
python server.py ~/checkpoints/sam2 large --port <your_port> --host <your_hostname> --baseline
```

Generate and save rles (one line per json via `-w "\n"`)
```
$ time xargs -I {} curl -s -w "\n" -X POST http://<your_hostname>:<your_port>/upload_rle -F 'image=@{}' < sav_val_image_paths_shuf_1000 > results/sav_val_masks_baseline_shuf_1000

real    13m6.374s
user    0m3.349s
sys     0m4.137s
```

### 3. Start server with torchao variant of SAM2
Start server
```
python server.py ~/checkpoints/sam2 large --port <your_port> --host <your_hostname>
```

Generate and save rles (one line per json via `-w "\n"`)
```
$ time xargs -I {} curl -s -w "\n" -X POST http://<your_hostname>:<your_port>/upload_rle -F 'image=@{}' < sav_val_image_paths_shuf_1000 > results/sav_val_masks_shuf_1000

real    12m18.916s
user    0m3.506s
sys     0m4.350s
```

### 4. Start server with torchao variant of SAM2 and `--fast` optimizations
Start server
```
python server.py ~/checkpoints/sam2 large --port <your_port> --host <your_hostname> --fast
```

Generate and save rles (one line per json via `-w "\n"`)
```
$ time xargs -I {} curl -s -w "\n" -X POST http://<your_hostname>:<your_port>/upload_rle -F 'image=@{}' < sav_val_image_paths_shuf_1000 > results/sav_val_masks_fast_shuf_1000

real    9m23.912s
user    0m3.271s
sys     0m4.138s
```

### 5. Start server with torchao variant of SAM2 and `--fast` and `--furious` optimizations
Start server
```
python server.py ~/checkpoints/sam2 large --port <your_port> --host <your_hostname> --fast --furious
```

Generate and save rles (one line per json via `-w "\n"`)
```
$ time xargs -I {} curl -s -w "\n" -X POST http://<your_hostname>:<your_port>/upload_rle -F 'image=@{}' < sav_val_image_paths_shuf_1000 > results/sav_val_masks_fast_furious_shuf_1000

real    3m24.383s
user    0m3.583s
sys     0m4.519s
```
