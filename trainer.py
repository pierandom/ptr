from time import perf_counter
from datetime import timedelta
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from einops import rearrange

from dataset import NextTokenDataset
from metrics import AverageMetric


class Trainer:
    def __init__(
        self,
        rank: int,
        world_size: int,
        ddp_model: DDP,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_dataset: NextTokenDataset,
        val_dataset: NextTokenDataset,
        config: Dict,
    ):
        self.rank = rank
        self.world_size = world_size
        self.ddp_model = ddp_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.scaler = GradScaler()
        self.train_loss_metric = AverageMetric(self.rank, self.world_size)
        self.iterations = 0
        self.tokens_processed = 0
        self.tot_train_time = timedelta(seconds=0)

    def _save_ckpt(self):
        state = {
            "model": self.ddp_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "iterations": self.iterations,
            "tokens_processed": self.tokens_processed,
            "tot_train_time": self.tot_train_time,
        }
        torch.save(state, self.config["checkpoint_path"])

    def _load_ckpt(self):
        ckpt = torch.load(self.config["checkpoint_path"], map_location=f"cuda:{self.rank}")
        self.ddp_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.iterations = ckpt["iterations"]
        self.tokens_processed = ckpt["tokens_processed"]
        self.tot_train_time = timedelta(seconds=ckpt["tot_train_time"])

    def _update_grads(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.ddp_model(inputs, is_causal=True)
            loss = F.cross_entropy(rearrange(logits, "b t c -> b c t"), targets)
            loss = loss / self.config["grad_accumulation_steps"]

        self.scaler.scale(loss).backward()
        self.train_loss_metric.update(loss.item())

    def _update_state(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=1)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

    def _compute_elapsed_time(
        self, start_grad_update_time: float, start_train_time: float
    ) -> Tuple[str, str]:
        """Compute time elapsed since last grad update and from the training start in milliseconds."""
        end_time = perf_counter()
        grad_update_time = timedelta(seconds=end_time - start_grad_update_time)
        train_time = (
            timedelta(seconds=end_time - start_train_time) + self._train_time
        )
        return str(grad_update_time)[:-3], str(train_time)[:-3]

    def _print_summary(
        self,
        start_grad_update_time: float,
        start_train_time: float,
    ):
        grad_update_time, train_time = self._compute_elapsed_time(
            start_grad_update_time, start_train_time
        )

        # metric compute method needs to be called by all processes
        train_loss = (
            self.train_loss_metric.compute() * self.config["grad_accumulation_steps"]
        )
        if self.rank == 0:
            print(
                f"Time: {grad_update_time}/{train_time} - "
                f"Updates: {self.iterations//self.config['grad_accumulation_steps']:,} - "
                f"Tokens processed: {self.tokens_processed:,} - "
                f"Train Loss: {train_loss:.6f} - "
                f"LR: {self.scheduler.get_last_lr()[0]:E}"
            )

    def train(self):

        start_train_time = start_grad_update_time = perf_counter()
        while True:
            for token_ids in self.train_dataset:
                self.iterations += 1
                self.optimizer.zero_grad()

                token_ids = token_ids.to(self.rank)
                inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
                self.tokens_processed += inputs.numel()

                if self.iterations % self.config["grad_accumulation_steps"] != 0:
                    with self.ddp_model.no_sync():
                        self._update_grads(inputs, targets)
                else:
                    self._update_grads(inputs, targets)
                    self._update_state()

                    self._print_summary(
                        start_grad_update_time,
                        start_train_time,
                    )

                if (
                    self.iterations
                    % (
                        self.config["save_every_n_steps"]
                        * self.config["grad_accumulation_steps"]
                    )
                    == 0
                    and self.rank == 0
                ):
                    print("Saving checkpoint...")
                    self._save_ckpt()

                if (
                    self.iterations
                    % (
                        self.config["validate_every_n_steps"]
                        * self.config["grad_accumulation_steps"]
                    )
                    == 0
                ):
                    if self.rank == 0:
                        print("Validating...")
                    start_val_time = perf_counter()
                    val_loss = self.validate()
                    val_time = str(
                        timedelta(seconds=perf_counter() - start_val_time())
                    )[:-3]
                    if self.rank == 0:
                        print(f"Time: {val_time} - Val Loss: {val_loss:.6f}")

                if self.iterations % self.config["grad_accumulation_steps"] == 0:
                    start_grad_update_time = perf_counter()

    @torch.no_grad()
    def validate(self) -> float:
        val_loss = AverageMetric(self.rank, self.world_size)
        for token_ids in self.val_dataset:
            token_ids = token_ids.to(self.rank)
            inputs, targets = token_ids[:, :-1], token_ids[:, 1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.ddp_model(inputs, is_causal=True)
                loss = F.cross_entropy(rearrange(logits, "b t c -> b c t"), targets)

            val_loss.update(loss.item())
        return val_loss.compute()
