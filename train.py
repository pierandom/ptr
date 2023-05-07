import os
from argparse import ArgumentParser
from time import perf_counter
from datetime import timedelta
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.cuda.amp import GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from einops import rearrange

from model import PTR
from dataset import NextTokenDataset
from metrics import AverageMetric


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--head_dim", type=int, default=32)
    parser.add_argument("--num_attention_layers", type=int, default=12)
    parser.add_argument("--ffn_factor", type=int, default=2)
    parser.add_argument("--context_len", type=int, default=513)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_acc_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero_optimizer", action="store_true")

    args = parser.parse_args()
    return args


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class Trainer:
    def __init__(
        self,
        rank: int,
        world_size: int,
        model: DDP,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_dataset: NextTokenDataset,
        val_dataset: NextTokenDataset,
        grad_acc_steps: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.grad_acc_steps = grad_acc_steps
        self.scaler = GradScaler()
        self.train_loss_metric = AverageMetric(self.rank, self.world_size)
        self.val_loss_metric = AverageMetric(self.rank, self.world_size)

    def _update_grads(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.model(inputs, is_causal=True)
            loss = F.cross_entropy(rearrange(logits, "b t c -> b c t"), targets)
            loss = loss / self.grad_acc_steps

        self.scaler.scale(loss).backward()
        self.train_loss_metric.update(loss.item())

    def _update_state(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

    def _print_summary(self, iterations: int, tokens_processed: int):
        end_time = perf_counter()
        # compute execution times in milliseconds
        train_time = str(timedelta(seconds=end_time - self.start_train_time))[:-3]
        grad_update_time = str(
            timedelta(seconds=end_time - self.start_grad_update_time)
        )[:-3]

        # metric compute method needs to be called by all processes
        train_loss = self.train_loss_metric.compute() * self.grad_acc_steps
        if self.rank == 0:
            print(
                f"Time: {grad_update_time}/{train_time} - "
                f"Updates: {iterations//self.grad_acc_steps:,} - Tokens processed: {tokens_processed:,} - "
                f"Train Loss: {train_loss:.6f} - LR: {self.scheduler.get_last_lr()[0]:E}"
            )

        self.start_grad_update_time = perf_counter()

    def train(self):
        torch.cuda.set_device(self.rank)
        tokens_processed = 0
        iterations = 0

        self.start_train_time = self.start_grad_update_time = perf_counter()
        while True:
            for token_ids in self.train_dataset:
                iterations += 1
                self.optimizer.zero_grad()

                token_ids = token_ids.to(self.rank)
                inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
                tokens_processed += inputs.numel()

                if iterations % self.grad_acc_steps != 0:
                    with self.model.no_sync():
                        self._update_grads(inputs, targets)
                else:
                    self._update_grads(inputs, targets)
                    self._update_state()

                    self._print_summary(iterations, tokens_processed)

    @torch.no_grad()
    def validate(
        rank: int, world_size: int, model: DDP, dataset: NextTokenDataset
    ) -> AverageMetric:
        val_loss = AverageMetric(rank, world_size)
        for token_ids in dataset:
            token_ids = token_ids.to(rank)
            inputs, targets = token_ids[:, :-1], token_ids[:, 1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs, is_causal=True)
                loss = F.cross_entropy(rearrange(logits, "b t c -> b c t"), targets)

            val_loss.update(loss.item())
        return val_loss


def main(rank, world_size, args):
    setup(rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained("tokenizer_ptr")

    # model
    model: torch.nn.Module = PTR(
        emb_dim=args.embedding_dim,
        vocab_size=tokenizer.vocab_size,
        head_dim=args.head_dim,
        num_attn_layers=args.num_attention_layers,
        ffn_factor=args.ffn_factor,
    ).to(rank)
    model = torch.compile(model)
    ddp_model = DDP(model, device_ids=[rank])

    # optimizer
    if args.zero_optimizer:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(), optimizer_class=torch.optim.AdamW, lr=args.learning_rate
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # lr_scheduler
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.ConstantLR(optimizer),
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=1000, T_mult=2
            ),
        ],
        milestones=[1000],
    )

    # datasets
    train_dataset = NextTokenDataset(
        split="train",
        rank=rank,
        world_size=world_size,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        context_len=args.context_len,
    )
    val_dataset = NextTokenDataset(
        split="val",
        rank=rank,
        world_size=world_size,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        context_len=args.context_len,
    )

    trainer = Trainer(
        rank=rank,
        world_size=world_size,
        model=ddp_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        grad_acc_steps=args.grad_acc_steps,
    )
    trainer.train()

    cleanup()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
