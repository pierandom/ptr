from argparse import ArgumentParser
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import datetime

from dataset import NextTokenDataset
from model import PTR
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


def main(rank, world_size, args, config, run_path):
    setup(rank, world_size, config)

    tokenizer = AutoTokenizer.from_pretrained("pierandom/ptr")

    train_dataset = NextTokenDataset(
        split="train",
        rank=rank,
        world_size=world_size,
        tokenizer=tokenizer,
        batch_size=config["dataset"]["batch_size"],
        context_len=config["dataset"]["context_len"],
    )
    val_dataset = NextTokenDataset(
        split="val",
        rank=rank,
        world_size=world_size,
        tokenizer=tokenizer,
        batch_size=config["dataset"]["batch_size"],
        context_len=config["dataset"]["context_len"],
    )

    model: torch.nn.Module = PTR(
        emb_dim=config["model"]["embedding_dim"],
        vocab_size=tokenizer.vocab_size,
        head_dim=config["model"]["head_dim"],
        num_attn_layers=config["model"]["num_attention_layers"],
        ffn_factor=config["model"]["ffn_factor"],
    )
    model = model.to(rank)
    model = torch.compile(model)
    ddp_model = DDP(model, device_ids=[rank])

    if config["optimizer"]["ZeRO"]:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=config["scheduler"]["lr"],
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["scheduler"]["lr"])

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.ConstantLR(optimizer),
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
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
        trainer._print("Loading checkpoint...")
        trainer.load_ckpt()

    trainer.train()

    cleanup()


if __name__ == "__main__":
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    run_path, config = init_run(args.resume_run_id)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(
        main, args=(WORLD_SIZE, args, config, run_path), nprocs=WORLD_SIZE, join=True
    )
