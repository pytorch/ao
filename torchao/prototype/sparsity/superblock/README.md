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
* Install PyTorch. For best performance, we recommend the pytorch nightlies
  ```
  pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
  ```
  We ran our experiments with torch==2.6.0.dev20240924+cu121


# Results

### Benchmarking
For all our benchmarking results, you can run `benchmark.sh`.
These benchmarks were run on a NVIDIA-A100-80GB, with cuSPARSELt v0.5.2.


### Evaluation

To reproduce our accuracy results, you can run `evaluate.sh`
You will need to set the following environment variables first to run the script:

```
IMAGENET_PATH=<put the path of ImageNet dataset here>
NGPUS=1 # put number of available GPUS here
```

## Training
Please refer to [TRAINING.md](TRAINING.md) for training from scratch. We use [Torchvision](https://github.com/pytorch/vision/tree/main/references/classification) as our framework for training. Supermask can be applied during training.

For example, if you would like to train a `vit_b_16` from scratch using Supermask, you can use the respective torchvision command found in [TRAINING.md](TRAINING.md) and append the supermask arguments:
```
torchrun --nproc_per_node=8 train.py\
    --model vit_h_14 --epochs 3 --batch-size 64 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 \
    --clip-grad-norm 1 --cutmix-alpha 1.0 --model-ema\
    --sparsity semi_structured --data-path $IMAGENET_PATH
```
Through this command, we are training a `vit_b_16` with 90% sparsity to linear layers using 32x32 tiles.

Please run `python train.py --help` for a full list of available arguments.


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
## License
SuperBlock is released under the [MIT license](https://github.com/pytorch-labs/superblock?tab=MIT-1-ov-file#readme).
