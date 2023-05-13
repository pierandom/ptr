from argparse import ArgumentParser
import os
import yaml
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from dataset import NextTokenDataset
from model import PTR
from trainer import Trainer

def parse_args():
    parser = ArgumentParser()

    args = parser.parse_args()
    return args


def setup(rank, world_size, config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    torch.cuda.manual_seed(config["seed"])


def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args, config):
    setup(rank, world_size, config)

    tokenizer = AutoTokenizer.from_pretrained("tokenizer_ptr")

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
        config=config["training"],
    )
    trainer.train()

    cleanup()


if __name__ == "__main__":
    args = parse_args()

    with open("config.yaml") as config_file:
        config = yaml.safe_load(config_file)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(main, args=(WORLD_SIZE, args, config), nprocs=WORLD_SIZE, join=True)
