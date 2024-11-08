## Example curl command
```
curl -X POST http://127.0.0.1:5000/upload -F 'image=@/path/to/file.jpg' --output path/to/output.png
```

## Example script to collect rles

Start the server

```
python server.py ~/checkpoints/sam2 --port <your_port> --host <your_hostname> --fast
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
|              ao | 1.0                |   0                 | 840                 |  4350MiB (4%)         |  1         |   64             |
|            fast | 0.9897813200950623 | 191                 | 661                 |  3916MiB (4%)         |  1         |   64             |
|            fast | 0.9897371530532837 | 192                 | 388                 | 50787MiB (52%)        | 16         | 1024             |
|  fast + furious | 0.974319338798523  | 209                 | 461                 |  3453MiB (3%)         |  1         |   64             |
|  fast + furious | 0.9702069759368896 | 196                 | 195                 | 48298MiB (49%)        | 16         | 1024             |

mask count mismatch counts the number of requests where the number of masks differ from the baseline.
For example, the baseline may have chosen to segment an image into 18 masks, but the fast variant produces 17 or 19.
We exclude these examples from the mIoU calculation.

The 'ao' mode is a copy of the baseline with modifications to make the code compile-able and improve the performance of fast.

### 0. Download checkpoints and install requirements

```
pip install -r requirements.txt
```

Download `sam2.1_hiera_large.pt` from https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints and put it into `~/checkpoints/sam2`

### 1. Create a random subset of 1000 images
```
find sav_val -type f > sav_val_image_paths
shuf -n 1000 sav_val_image_paths > sav_val_image_paths_shuf_1000
```

### 2. Use the baseline (https://github.com/facebookresearch/sam2) to generate rles

Make sure you've installed https://github.com/facebookresearch/sam2

Start server
```
python server.py ~/checkpoints/sam2 --port <your_port> --host <your_hostname> --baseline
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
python server.py ~/checkpoints/sam2 --port <your_port> --host <your_hostname>
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
python server.py ~/checkpoints/sam2 --port <your_port> --host <your_hostname> --fast
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
python server.py ~/checkpoints/sam2 --port <your_port> --host <your_hostname> --fast --furious
```

Generate and save rles (one line per json via `-w "\n"`)
```
$ time xargs -I {} curl -s -w "\n" -X POST http://<your_hostname>:<your_port>/upload_rle -F 'image=@{}' < sav_val_image_paths_shuf_1000 > results/sav_val_masks_fast_furious_shuf_1000

real    3m24.383s
user    0m3.583s
sys     0m4.519s
```
