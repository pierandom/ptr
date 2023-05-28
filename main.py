from argparse import ArgumentParser
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import datetime

from dataset import NextTokenDataset
from model import PTR, PTRConfig
from trainer import Trainer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--resume_run_id", type=str, help="Run id to resume")

    args = parser.parse_args()
    return args


def init_run(run_id: Optional[str]) -> Tuple[Path, Dict]:
    runs_dir = Path("runs")
    if run_id:
        run_path = runs_dir / run_id
    else:
        now = datetime.datetime.now()
        run_path = runs_dir / now.strftime("%Y%m%d_%H%M%S")

    run_path.mkdir(parents=True, exist_ok=True)

    if not run_id:
        shutil.copy("config.yaml", str(run_path))

    config_path = run_path / "config.yaml"
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)

    return run_path, config


def setup(rank, world_size, config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    torch.cuda.manual_seed(config["seed"])


def cleanup():
    dist.destroy_process_group()


def num_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _main(rank, world_size, args, config, run_path):
    setup(rank, world_size, config)

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])

    train_dataset = NextTokenDataset(
        split="train",
        rank=rank,
        world_size=world_size,
        tokenizer=tokenizer,
        batch_size=config["dataset"]["batch_size"],
        context_len=config["dataset"]["context_len"],
        shuffle=True,
    )
    val_dataset = NextTokenDataset(
        split="val",
        rank=rank,
        world_size=world_size,
        tokenizer=tokenizer,
        batch_size=config["dataset"]["batch_size"],
        context_len=config["dataset"]["context_len"],
        max_examples=config["dataset"]["max_validation_examples"],
    )

    model_config = PTRConfig(
        emb_dim=config["model"]["embedding_dim"],
        vocab_size=tokenizer.vocab_size,
        head_dim=config["model"]["head_dim"],
        num_attn_layers=config["model"]["num_attention_layers"],
        ffn_factor=config["model"]["ffn_factor"],
    )
    model = PTR(model_config)

    model = model.to(rank)
    model = torch.compile(model)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(ddp_model.parameters(), lr=config["scheduler"]["lr"])

    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        [
            lr_scheduler.ConstantLR(optimizer),
            lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config["scheduler"]["base_period"],
                T_mult=config["scheduler"]["period_factor"],
            ),
        ],
        milestones=[config["scheduler"]["warmup_steps"]],
    )

    trainer = Trainer(
        rank=rank,
        world_size=world_size,
        ddp_model=ddp_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        run_path=run_path,
        config=config["training"],
    )

    if args.resume_run_id:
        trainer.print("Loading checkpoint...")
        trainer.load_ckpt()
    else:
        trainer.print(f"# Parameters: {num_parameters(model):,}")

    trainer.train()


def main(rank, *args, **kwargs):
    try:
        _main(rank, *args, **kwargs)
    except KeyboardInterrupt:
        print(f"Catching KeyboardInterrupt on process {rank}. Terminating...")
    finally:
        cleanup()


if __name__ == "__main__":
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    run_path, config = init_run(args.resume_run_id)

    WORLD_SIZE = torch.cuda.device_count()
    try:
        mp.spawn(
            main,
            args=(WORLD_SIZE, args, config, run_path),
            nprocs=WORLD_SIZE,
            join=True,
        )
    except KeyboardInterrupt:
        print(f"Catching KeyboardInterrupt on main process. Terminating...")
