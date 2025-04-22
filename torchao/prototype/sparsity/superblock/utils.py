#  Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import copy
import datetime
import errno
import hashlib
import math
import os
import time
from collections import OrderedDict, defaultdict, deque
from typing import List, Optional, Tuple

import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

from torchao.prototype.sparsity.sparsifier.weight_norm_sparsifier import (
    WeightNormSparsifier,
)
from torchao.prototype.sparsity.superblock.blocksparse import block_sparse_weight
from torchao.prototype.sparsity.superblock.supermask import (
    SupermaskLinear,
    apply_supermask,
)
from torchao.quantization import int8_dynamic_activation_int8_weight, quantize_
from torchao.sparsity import semi_sparse_weight, sparsify_


def get_args_parser(train=False, evaluate=False, benchmark=False):
    assert sum([train, evaluate, benchmark]) == 1, (
        "One and only one of training, evaluation, or benchmark can be true"
    )

    # Shared common args
    parser = argparse.ArgumentParser(
        description="SuperBlock Imagenet Training/Evaluation/Benchmarking Script",
        add_help=True,
    )
    parser.add_argument("--data-path", type=str, help="IMAGENET dataset path")
    parser.add_argument(
        "--model",
        default="vit_b_16",
        choices=["vit_b_16", "vit_h_14"],
        type=str,
        help="ViT base model",
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="device (Default: cuda)"
    )
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="per device batch size"
    )
    parser.add_argument(
        "--val-crop-size",
        default=224,
        type=int,
        help="the central crop size used for validation (default: 224)",
    )
    parser.add_argument(
        "--sparsity",
        choices=["bsr", "semi_structured"],
        default=None,
        help="weight sparsification to apply",
    )
    parser.add_argument(
        "--bsr",
        type=int,
        nargs="?",
        const=256,
        default=None,
        help="Convert sparsified weights to BSR format with optional block size (default: 256)",
    )
    parser.add_argument("--sparsity-linear", type=float, default=0.0)
    parser.add_argument("--sparsity-conv1x1", type=float, default=0.0)
    parser.add_argument("--sparsity-conv", type=float, default=0.0)
    parser.add_argument(
        "--skip-last-layer-sparsity",
        action="store_true",
        help="Skip applying sparsity to the last linear layer (for vit only)",
    )
    parser.add_argument(
        "--skip-first-transformer-sparsity",
        action="store_true",
        help="Skip applying sparsity to the first transformer layer (for vit only)",
    )
    parser.add_argument(
        "--quantization", action="store_true", help="Run with int8 dynamic quantization"
    )
    parser.add_argument(
        "--weights", default=None, type=str, help="the weights enum name to load"
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        help="optional checkpoint to load weights after intialization",
    )
    parser.add_argument(
        "--header", action="store_true", help="Print header for first run"
    )

    # Eval a subset of training args
    # lots of training args
    if train or evaluate:
        parser.add_argument(
            "-j",
            "--workers",
            default=16,
            type=int,
            metavar="N",
            help="number of data loading workers",
        )
        parser.add_argument(
            "--accumulation-steps",
            default=1,
            type=int,
            help="Number of steps to accumulate gradients over",
        )
        parser.add_argument(
            "--epochs",
            default=90,
            type=int,
            metavar="N",
            help="number of total epochs to run",
        )
        parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
        parser.add_argument(
            "--lr", default=0.1, type=float, help="initial learning rate"
        )
        parser.add_argument(
            "--momentum", default=0.9, type=float, metavar="M", help="momentum"
        )
        parser.add_argument(
            "--wd",
            "--weight-decay",
            default=1e-4,
            type=float,
            metavar="W",
            help="weight decay",
            dest="weight_decay",
        )
        parser.add_argument(
            "--norm-weight-decay",
            default=None,
            type=float,
            help="weight decay for Normalization layers (default: None, same value as --wd)",
        )
        parser.add_argument(
            "--bias-weight-decay",
            default=None,
            type=float,
            help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
        )
        parser.add_argument(
            "--transformer-embedding-decay",
            default=None,
            type=float,
            help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
        )
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            help="label smoothing (default: 0.0)",
            dest="label_smoothing",
        )
        parser.add_argument(
            "--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)"
        )
        parser.add_argument(
            "--cutmix-alpha",
            default=0.0,
            type=float,
            help="cutmix alpha (default: 0.0)",
        )
        parser.add_argument(
            "--lr-scheduler",
            default="steplr",
            type=str,
            help="the lr scheduler (default: steplr)",
        )
        parser.add_argument(
            "--lr-warmup-epochs",
            default=0,
            type=int,
            help="the number of epochs to warmup (default: 0)",
        )
        parser.add_argument(
            "--lr-warmup-method",
            default="constant",
            type=str,
            help="the warmup method (default: constant)",
        )
        parser.add_argument(
            "--lr-warmup-decay", default=0.01, type=float, help="the decay for lr"
        )
        parser.add_argument(
            "--lr-step-size",
            default=30,
            type=int,
            help="decrease lr every step-size epochs",
        )
        parser.add_argument(
            "--lr-gamma",
            default=0.1,
            type=float,
            help="decrease lr by a factor of lr-gamma",
        )
        parser.add_argument(
            "--lr-min",
            default=0.0,
            type=float,
            help="minimum lr of lr schedule (default: 0.0)",
        )
        parser.add_argument(
            "--print-freq", default=10, type=int, help="print frequency"
        )
        parser.add_argument(
            "--output-dir", default=".", type=str, help="path to save outputs"
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help='Resumes training from latest available checkpoint ("model_<epoch>.pth")',
        )
        parser.add_argument(
            "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
        )
        parser.add_argument(
            "--cache-dataset",
            dest="cache_dataset",
            help="Cache the datasets for quicker initialization. It also serializes the transforms",
            action="store_true",
        )
        parser.add_argument(
            "--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true"
        )
        parser.add_argument(
            "--auto-augment",
            default=None,
            type=str,
            help="auto augment policy (default: None)",
        )
        parser.add_argument(
            "--ra-magnitude",
            default=9,
            type=int,
            help="magnitude of auto augment policy",
        )
        parser.add_argument(
            "--augmix-severity", default=3, type=int, help="severity of augmix policy"
        )
        parser.add_argument(
            "--random-erase",
            default=0.0,
            type=float,
            help="random erasing probability (default: 0.0)",
        )
        # Mixed precision training parameters
        parser.add_argument(
            "--amp",
            action="store_true",
            help="Use torch.cuda.amp for mixed precision training",
        )
        # distributed training parameters
        parser.add_argument(
            "--world-size", default=1, type=int, help="number of distributed processes"
        )
        parser.add_argument(
            "--dist-url",
            default="env://",
            type=str,
            help="url used to set up distributed training",
        )
        parser.add_argument(
            "--model-ema",
            action="store_true",
            help="enable tracking Exponential Moving Average of model parameters",
        )
        parser.add_argument(
            "--model-ema-steps",
            type=int,
            default=32,
            help="the number of iterations that controls how often to update the EMA model (default: 32)",
        )
        parser.add_argument(
            "--model-ema-decay",
            type=float,
            default=0.99998,
            help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
        )
        parser.add_argument(
            "--use-deterministic-algorithms",
            action="store_true",
            help="Forces the use of deterministic algorithms only.",
        )
        parser.add_argument(
            "--interpolation",
            default="bilinear",
            type=str,
            help="the interpolation method (default: bilinear)",
        )
        parser.add_argument(
            "--val-resize-size",
            default=256,
            type=int,
            help="the resize size used for validation (default: 256)",
        )
        parser.add_argument(
            "--train-crop-size",
            default=224,
            type=int,
            help="the random crop size used for training (default: 224)",
        )
        parser.add_argument(
            "--clip-grad-norm",
            default=None,
            type=float,
            help="the maximum gradient norm (default None)",
        )
        parser.add_argument(
            "--ra-reps",
            default=3,
            type=int,
            help="number of repetitions for Repeated Augmentation (default: 3)",
        )
        parser.add_argument(
            "--meta", action="store_true", help="Use Meta internal imagenet structure"
        )

    if benchmark:
        parser.add_argument(
            "--dtype",
            choices=["float32", "bfloat16", "float16"],
            help="Data type",
            default="bfloat16",
        )
        parser.add_argument(
            "--tune-kernel-params",
            action="store_true",
            help="Tune kernel params for BSR",
        )
        parser.add_argument(
            "--profile", action="store_true", help="Dump Prefetto trace"
        )

    return parser


# filter functions
def mlp_0_only(mod, name):
    return isinstance(mod, torch.nn.Linear) and "mlp.0" in name


def mlp_3_only(mod, name):
    return isinstance(mod, torch.nn.Linear) and "mlp.3" in name


def mlp_only(mod, name):
    return isinstance(mod, torch.nn.Linear) and "mlp" in name


def superblock_only(mod, name):
    return isinstance(mod, SupermaskLinear) and "mlp" in name


def mlp_only_with_args(
    mod, name, skip_last_layer_sparsity=False, skip_first_transformer_sparsity=False
):
    if skip_last_layer_sparsity and "heads.head" in name:
        return False
    if skip_first_transformer_sparsity and "encoder.layers.encoder_layer_0" in name:
        return False
    if isinstance(mod, torch.nn.Linear) and "mlp" in name:
        return True
    return False


### Custom sparsification utils
def apply_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, SupermaskLinear) and "mlp" in name:
            module.sparsify_offline()


def accelerate_with_sparsity(model, args):
    if args.sparsity == "bsr":
        apply_sparsity(model)
        if args.quantization:
            from torchao.dtypes import BlockSparseLayout

            quantize_(
                model,
                int8_dynamic_activation_int8_weight(
                    _layout=BlockSparseLayout(blocksize=args.bsr)
                ),
                superblock_only,
            )
        else:
            assert args.bsr is not None, "BSR requires a block size"
            sparsify_(model, block_sparse_weight(blocksize=args.bsr), superblock_only)
    elif args.sparsity == "semi_structured":
        if args.quantization:
            from torchao.dtypes import SemiSparseLayout

            quantize_(
                model,
                int8_dynamic_activation_int8_weight(layout=SemiSparseLayout()),
                mlp_0_only,
            )
            sparsify_(model, semi_sparse_weight(), mlp_3_only)
        else:
            sparsify_(model, semi_sparse_weight(), mlp_only)
    else:
        if args.quantization:
            quantize_(model, int8_dynamic_activation_int8_weight(), mlp_only)


def simulate_sparsity(model, args):
    if args.sparsity == "bsr":
        apply_supermask(
            model,
            linear_sparsity=args.sparsity_linear,
            linear_sp_tilesize=args.bsr,
            conv1x1_sparsity=args.sparsity_conv1x1,
            conv1x1_sp_tilesize=args.bsr,
            conv_sparsity=args.sparsity_conv,
            conv_sp_tilesize=args.bsr,
            skip_last_layer_sparsity=args.skip_last_layer_sparsity,
            skip_first_transformer_sparsity=args.skip_first_transformer_sparsity,
            device=args.device,
            verbose=False,
        )
    elif args.sparsity == "semi_structured":
        sparse_config = []
        for name, mod in model.named_modules():
            if mlp_only_with_args(
                mod,
                name,
                skip_first_transformer_sparsity=args.skip_first_transformer_sparsity,
                skip_last_layer_sparsity=args.skip_last_layer_sparsity,
            ):
                sparse_config.append({"tensor_fqn": f"{name}.weight"})

        sparsifier = WeightNormSparsifier(
            sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
        )
        sparsifier.prepare(model, sparse_config)
        sparsifier.step()
        return sparsifier


# ------------------------------------------------------------
# The following code contains torchvision reference code,
# largely copied from: https://github.com/pytorch/vision/tree/main/references/classification
# Please open issues in the original repository if you have questions.


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank})", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16

    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                f"For checkpoint {f}, expected list of params: {params_keys}, but found: {model_params_keys}"
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def store_model_weights(model, checkpoint_path, checkpoint_key="model", strict=True):
    """
    This method can be used to prepare weights files for new models. It receives as
    input a model architecture and a checkpoint from the training script and produces
    a file with the weights ready for release.

    Examples:
        from torchvision import models as M

        # Classification
        model = M.mobilenet_v3_large(weights=None)
        print(store_model_weights(model, './class.pth'))

        # Quantized Classification
        model = M.quantization.mobilenet_v3_large(weights=None, quantize=False)
        model.fuse_model(is_qat=True)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        _ = torch.ao.quantization.prepare_qat(model, inplace=True)
        print(store_model_weights(model, './qat.pth'))

        # Object Detection
        model = M.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
        print(store_model_weights(model, './obj.pth'))

        # Segmentation
        model = M.segmentation.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True)
        print(store_model_weights(model, './segm.pth', strict=False))

    Args:
        model (pytorch.nn.Module): The model on which the weights will be loaded for validation purposes.
        checkpoint_path (str): The path of the checkpoint we will load.
        checkpoint_key (str, optional): The key of the checkpoint where the model weights are stored.
            Default: "model".
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        output_path (str): The location where the weights are saved.
    """
    # Store the new model next to the checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Deep copy to avoid side-effects on the model object.
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load the weights to the model to validate that everything works
    # and remove unnecessary weights (such as auxiliaries, etc)
    if checkpoint_key == "model_ema":
        del checkpoint[checkpoint_key]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            checkpoint[checkpoint_key], "module."
        )
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    torch.distributed.barrier()
    torch.distributed.all_reduce(t)
    return t


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = (
                    f"{prefix}.{name}" if prefix != "" and "." in key else name
                )
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append(
                {"params": params[key], "weight_decay": params_weight_decay[key]}
            )
    return param_groups


# Presets for ImageNet training/eval taken from: https://github.com/pytorch/vision/blob/main/references/classification/presets.py


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(
                    autoaugment.RandAugment(
                        interpolation=interpolation, magnitude=ra_magnitude
                    )
                )
            elif auto_augment_policy == "ta_wide":
                trans.append(
                    autoaugment.TrivialAugmentWide(interpolation=interpolation)
                )
            elif auto_augment_policy == "augmix":
                trans.append(
                    autoaugment.AugMix(
                        interpolation=interpolation, severity=augmix_severity
                    )
                )
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(
                    autoaugment.AutoAugment(
                        policy=aa_policy, interpolation=interpolation
                    )
                )
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):
        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


# transforms taken from: https://github.com/pytorch/vision/blob/main/references/classification/transforms.py


class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(
        self, batch: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(
                "Please provide a valid positive value for the num_classes."
            )
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(
        self, batch: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        _, H, W = F.get_dimensions(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


# RA Sampler implementaion taken from: https://github.com/pytorch/vision/blob/main/references/classification/sampler.py


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(
        self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3
    ):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)
        )
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
