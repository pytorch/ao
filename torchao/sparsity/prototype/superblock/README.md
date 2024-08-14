# SuperBlock

SuperBlock combines two techniques for efficient neural network training and inference: Supermask and Block Compressed Sparse Row (BSR). 
The techniques are described in this [blog post](https://pytorch.org/blog/speeding-up-vits/).

### Supermask
[Supermask](https://arxiv.org/abs/2207.00670) is a technique for applying structured sparsity to neural networks using a learned mask. It works by learning a continuous mask (scores) that is applied element-wise to the weights of a neural network layer. The mask scores are learned separately from the weights and are thresholded based on a target sparsity level to obtain a binary mask. The mask determines which weigths are kept and which are pruned, and is learned during training.

During inference, the binary mask is applied element-wise to the weights, pruning the weights that correspond to a 0 in the mask, resulting in a sparse network that can be efficiently computed. 

### Block compressed Sparse Row Format (BSR)
[The BSR format](https://pytorch.org/docs/main/sparse.html#sparse-bsr-tensor) is a sparse matrix representation that stores dense sub-blocks of non-zero elements instead of individual non-zero elements. The matrix is divided into equal-sized blocks, and only the non-zero blocks are stored.

The BSR format is efficient for sparse matrices with a block structure, where non-zero elements tend to cluster in dense sub-blocks. It reduces storage requirements and enables efficient matrix operations on the non-zero blocks.

Currently, the BSR format is optimized for Nvidia A100 GPU(s) only.

## Setup
To use SuperBlock, you will need
* [PyTorch](https://pytorch.org/get-started/locally/)

To train the model or evaluate accuracy, you will need:
* ImageNet2012-blurred dataset

At least one GPU:
* A100 or H100

## Installation
* Clone this repo
  ```
  git clone https://github.com/pytorch-labs/superblock.git
  cd superblock
  ```
* Create a new conda environment
  ```
  conda create -n superblock
  conda activate superblock
  ```
* Install PyTorch. For best performance, we recommend `2.3.0.dev20240305+cu121` nightly
  ```
  pip install --pre torch==2.3.0.dev20240305+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121
  pip install --pre torchvision==0.18.0 --no-deps
  ```


## Benchmarking
Baseline:
```
python benchmark.py \
  --model vit_b_16 \
  --batch-size 256 \
  > /dev/null
```
Result:
```
532.1160546875 ms
```


80% sparsity, block size 64 (random weights):
```
python benchmark.py --model vit_b_16 \
  --batch-size 256 \
  --sparsity-linear 0.8 \
  --sp-linear-tile-size 64 \
  --sparsify-weights \
  --bsr 64 \
  > /dev/null
```
Result:
```
393.864453125 ms
```


## Training
Please refer to [TRAINING.md](TRAINING.md) for training from scratch. We use [Torchvision](https://github.com/pytorch/vision/tree/main/references/classification) as our framework for training. Supermask can be applied during training.

To apply supermask, we have the following arguments at our disposal,

* Apply Supermask to linear layers:
    ```
    --sparsity-linear
    --sp-linear-tile-size
    ```
* Apply Supermask to conv1x1 layers:
    ```
    --sparsity-conv1x1
    --sp-conv1x1-tile-size
    ```
* Apply Supermask to all other convolutional layers:
    ```
    --sparsity-conv
    --sp-conv-tile-size
    ```
* Skip the first transformer layer and/or last linear layer (ViT only):
    ```
    --skip-last-layer-sparsity
    --skip-first-transformer-sparsity
    ```

For example, if you would like to train a `vit_b_16` from scratch using Supermask, you can use the respective torchvision command found in [TRAINING.md](TRAINING.md) and append the supermask arguments:
```
torchrun --nproc_per_node=8 train.py\
    --model vit_b_16 --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra\
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema\ 
    --sparsity-linear 0.9 --sp-linear-tile-size 32
```
Through this command, we are training a `vit_b_16` with 90% sparsity to linear layers using 32x32 tiles.

Please run `python train.py --help` for a full list of available arguments.

## Evaluation

To run an evaluation of a Supermask-trained model, you can use [evaluate.py](evaluate.py). Our current version has signficant speedup with float32 only and not float16, hence, to illustrate speedup, we don't pass `--amp` in the example commands below.

```
MODEL_PATH=<put the path of the trained checkpoint here>
IMAGENET_PATH=<put the path of ImageNet dataset here>
NGPUS=1 # put number of available GPUS here
```

* Offline sparsification with BSR:
  ```
  torchrun --nproc_per_node=${NGPUS} evaluate.py  --model vit_b_16 --batch-size 256 --sparsity-linear 0.9 --sp-linear-tile-size 32 --weights-path ${MODEL_PATH}  --data-path ${IMAGENET_PATH} --sparsify-weights --bsr 32
  ```
  This command applies 90% sparsity to linear layers using 32x32 tiles, loads the model weights from ${MODEL_PATH}, loads the ImageNet validation set located at the specified path, applies offline sparsification to the weights, and converts the sparse weights to BSR format with a block size of 32. It is recommended to set `--bsr`      the same as tile size.

* Online sparsification without BSR:
  ```
  torchrun --nproc_per_node=${NGPUS} evaluate.py --model vit_b_16 --batch-size 256 --sparsity-linear 0.9 --sp-linear-tile-size 32 --weights-path ${MODEL_PATH} --data-path ${IMAGENET_PATH}
  ```
  This is similar to the previous command, but it does not apply offline sparsification or BSR conversion. Instead, the sparsity is applied on-the-fly during evaluation.

Please run `python evaluate.py --help` for a full list of available arguments.

Results (1x A100):
* Baseline
  ```
  Test:  Total time: 0:02:11
  Test:  Acc@1 78.392 Acc@5 93.592
  ```

* Sparsity= 0.9, Tile Size = 32, Online Sparsification, BSR = None
  ```
  Test:  Total time: 0:01:52
  Test:  Acc@1 76.092 Acc@5 92.656
  ```

* Sparsity= 0.9, Tile Size = 32, Offline Sparsification, BSR = None
  ```
  Test:  Total time: 0:01:54
  Test:  Acc@1 76.092 Acc@5 92.656
  ```

* Sparsity= 0.9, Tile Size = 32, Offline Sparsification, BSR = 32
  ```
  Test:  Total time: 0:01:25
  Test:  Acc@1 76.092 Acc@5 92.656
  ```

## Pretrained Weights

### Download:
Instead of training from scratch, if you'd like to use the Supermask weights of `vit_b_16` trained on privacy mitigated Imagenet-blurred, you can download them here:
```
SPARSITY=0.80 # Checkpoints available for: 0.70, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90
BLOCK_SIZE=32 # Checkpoints available for: 16, 32, 64
```

```
mkdir checkpoints
# For baseline,
wget https://huggingface.co/facebook/superblock-vit-b-16/resolve/main/checkpoints/baseline.pth -P checkpoints/
# For sparsified checkpoints,
wget https://huggingface.co/facebook/superblock-vit-b-16/resolve/main/checkpoints/sp${SPARSITY}-ts${BLOCK_SIZE}.pth -P checkpoints/
```

### Benchmark:
```
python benchmark.py --model vit_b_16 \
  --batch-size 256 \
  --sparsity-linear ${SPARSITY} \
  --sp-linear-tile-size ${BLOCK_SIZE} \
  --sparsify-weights \
  --bsr ${BLOCK_SIZE} \
  --weights-path ./checkpoints/sp${SPARSITY}-ts${BLOCK_SIZE}.pth \
  > /dev/null
```
Result:
```
530.342578125 ms
```

### Evaluate:
8 x A100 GPUs:
```
torchrun --nproc_per_node=8 evaluate.py --model vit_b_16 --batch-size 256 --sparsity-linear ${SPARSITY} --sp-linear-tile-size ${BLOCK_SIZE} --bsr ${BLOCK_SIZE} --sparsify-weights --weights-path checkpoints/sp${SPARSITY}-ts${BLOCK_SIZE}.pth --data-path ${IMAGENET_PATH}
```
Result:
```
Test:  Total time: 0:01:01
Test:  Acc@1 77.644 Acc@5 93.554
```

1 x A100 GPUs:
```
torchrun --nproc_per_node=1 evaluate.py --model vit_b_16 --batch-size 256 --sparsity-linear ${SPARSITY} --sp-linear-tile-size ${BLOCK_SIZE} --bsr ${BLOCK_SIZE} --sparsify-weights --weights-path checkpoints/sp${SPARSITY}-ts${BLOCK_SIZE}.pth --data-path ${IMAGENET_PATH}
```
Result:
```
Test:  Total time: 0:01:51
Test:  Acc@1 77.644 Acc@5 93.554
```

## License
SuperBlock is released under the [MIT license](https://github.com/pytorch-labs/superblock?tab=MIT-1-ov-file#readme).
