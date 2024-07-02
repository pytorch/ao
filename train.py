# pip install timm wandb tqdm datasets

import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import bitsandbytes as bnb
import datasets
import timm
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from torchao.prototype.optim_8bit import AdamDTQ8bit


class CosineSchedule:
    def __init__(self, lr: float, total_steps: int, warmup: float = 0.05) -> None:
        self.lr = lr
        self.final_lr = 0
        self.total_steps = total_steps
        self.warmup_steps = round(total_steps * warmup)

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * step / self.warmup_steps
        if step < self.total_steps:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.final_lr + 0.5 * (self.lr - self.final_lr) * (1 + math.cos(progress * math.pi))
        return self.final_lr


class WandbLogger:
    def __init__(self, args):
        if args.project is not None and not args.profile:
            import wandb

            Path("wandb_logs").mkdir(exist_ok=True)
            self.run = wandb.init(project=args.project, name=args.run_name, config=args, dir="wandb_logs")

        else:
            self.run = None

    def log(self, *args, **kwargs):
        if self.run is not None:
            self.run.log(*args, **kwargs)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)

    parser.add_argument("--amp", default="none")
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=4)

    parser.add_argument("--optim", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--cosine_lr_scheduler", action="store_true")

    parser.add_argument("--project")
    parser.add_argument("--run_name", default="debug")
    parser.add_argument("--profile", action="store_true")
    return parser


def get_dloader(args, training: bool):
    transforms = [v2.ToImage()]

    if training:
        transforms.extend([v2.RandomResizedCrop(224), v2.RandomHorizontalFlip()])
    else:
        transforms.extend([v2.Resize(256), v2.CenterCrop(224)])

    transforms.append(v2.ToDtype(torch.float32, scale=True))
    transforms.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transforms = v2.Compose(transforms)

    # use dataset from HF so download is fast
    ds = datasets.load_dataset("timm/resisc45", split="train" if training else "validation")
    ds = ds.select_columns(["image", "label"])
    ds.set_transform(lambda x: dict(image=transforms(x["image"]), label=x["label"]))

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.n_workers,
        pin_memory=training,
        drop_last=training,
    )


def get_amp_ctx(amp):
    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, none=None)[amp]
    return torch.autocast("cuda", dtype=dtype, enabled=amp != "none")


@torch.no_grad()
def evaluate_model(model, args):
    model.eval()
    val_dloader = get_dloader(args, False)

    all_labels = []
    all_preds = []

    for batch in tqdm(val_dloader, dynamic_ncols=True, desc=f"Evaluating"):
        all_labels.append(batch["label"].clone())
        if args.channels_last:
            batch["image"] = batch["image"].to(memory_format=torch.channels_last)

        with get_amp_ctx(args.amp):
            all_preds.append(model(batch["image"].cuda()).argmax(1).cpu())

    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    acc = (all_labels == all_preds).float().mean()
    return acc


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.profile:
        args.n_epochs = 1

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # wandb is only enabled when args.project is set and args.profile is False
    logger = WandbLogger(args)
    dloader = get_dloader(args, True)
    print(f"Train dataset: {len(dloader.dataset):,} images")

    model = timm.create_model(args.model, pretrained=True, num_classes=45).cuda()
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    if args.compile:
        model.compile(fullgraph=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    OPTIM_MAP = dict(
        Adam=torch.optim.Adam,
        AdamBnb8bit=bnb.optim.Adam8bit,
        AdamDTQ8bit=AdamDTQ8bit,
    )
    optim = OPTIM_MAP[args.optim](model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_schedule = CosineSchedule(args.lr, len(dloader) * args.n_epochs)

    grad_scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")

    step = 0
    for epoch_idx in range(args.n_epochs):
        model.train()
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if args.profile else nullcontext()

        with prof:
            for batch in tqdm(dloader, dynamic_ncols=True, desc=f"Epoch {epoch_idx + 1}/{args.n_epochs}"):
                if args.channels_last:
                    batch["image"] = batch["image"].to(memory_format=torch.channels_last)

                with get_amp_ctx(args.amp):
                    loss = F.cross_entropy(model(batch["image"].cuda()), batch["label"].cuda())
                grad_scaler.scale(loss).backward()

                if args.cosine_lr_scheduler:
                    lr = lr_schedule.get_lr(step)
                    for param_group in optim.param_groups:
                        param_group["lr"] = lr

                if step % 100 == 0:
                    logger.log(dict(loss=loss.item(), lr=optim.param_groups[0]["lr"]), step=step)

                grad_scaler.step(optim)
                grad_scaler.update()
                optim.zero_grad()

                step += 1

                if args.profile and step == 50:
                    break

        if args.profile:
            prof.export_chrome_trace("trace.json")

        else:
            val_acc = evaluate_model(model, args)
            print(f"Epoch {epoch_idx + 1}/{args.n_epochs}: val_acc={val_acc.item() * 100:.2f}")
            logger.log(dict(val_acc=val_acc), step=step)

    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1 << 30):.2f} GB")
