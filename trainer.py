import csv
from pathlib import Path
from time import perf_counter
from datetime import timedelta
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import torch.distributed as dist
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
        run_path: Path,
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
        self.iterations = 0
        self.tokens_processed = 0
        self.tot_train_time = timedelta(seconds=0)
        self.run_path = run_path
        self.path = {
            "checkpoint": run_path / "checkpoint.pth",
            "predictions": run_path / "predictions.txt",
            "output": run_path / "output.txt",
        }
        metrics_path = run_path / "metrics"
        metrics_path.mkdir(exist_ok=True)
        self.train_loss_metric = AverageMetric(
            self.rank, self.world_size, log_path=metrics_path / "train_loss.csv"
        )
        self.val_loss_metric = AverageMetric(
            self.rank, self.world_size, log_path=metrics_path / "val_loss.csv"
        )

    @property
    def grad_steps(self):
        return self.iterations / self.config["grad_accumulation_steps"]

    def print(self, string: str):
        if self.rank == 0:
            print(string)
            with self.path["output"].open("a") as out_file:
                out_file.write(string + "\n")

    def save_ckpt(self):
        if self.rank == 0:
            train_time = timedelta(seconds=perf_counter() - self.start_train_time)
            state = {
                "model": self.ddp_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "train_dataset": self.train_dataset.state_dict(),
                "train_loss_metric": self.train_loss_metric.state(),
                "iterations": self.iterations,
                "tokens_processed": self.tokens_processed,
                "tot_train_time": self.tot_train_time.seconds + train_time.seconds,
            }
            torch.save(state, self.path["checkpoint"])

    def load_ckpt(self):
        ckpt = torch.load(self.path["checkpoint"], map_location=f"cuda:{self.rank}")
        self.ddp_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.train_dataset.load_state_dict(ckpt["train_dataset"])
        self.train_loss_metric.load_state(ckpt["train_loss_metric"])
        self.iterations = ckpt["iterations"]
        self.tokens_processed = ckpt["tokens_processed"]
        self.tot_train_time = timedelta(seconds=ckpt["tot_train_time"])
        dist.barrier()

    def _update_grads(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.ddp_model(inputs, is_causal=True)
            loss = self.ddp_model.module.loss(
                logits, targets, self.config["use_entropy_weights"]
            )
            loss = loss / self.config["grad_accumulation_steps"]

        self.scaler.scale(loss).backward()
        self.train_loss_metric.update(
            loss.item() * self.config["grad_accumulation_steps"]
        )

    def _update_state(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.ddp_model.parameters(), max_norm=self.config["max_grad_norm"]
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

    def _compute_elapsed_time(self) -> Tuple[str, str]:
        """Compute time elapsed since last grad update and from the training start in milliseconds."""
        end_time = perf_counter()
        grad_update_time = timedelta(seconds=end_time - self.start_grad_update_time)
        train_time = (
            timedelta(seconds=end_time - self.start_train_time) + self.tot_train_time
        )
        return str(grad_update_time)[:-3], str(train_time)[:-3]

    def _summary(self):
        grad_update_time, train_time = self._compute_elapsed_time()

        # metric compute method needs to be called by all processes
        train_loss = self.train_loss_metric.compute()
        summary = (
            f"Time: {grad_update_time}/{train_time} - "
            f"Updates: {int(self.grad_steps):,} - "
            f"Tokens processed: {self.tokens_processed:,} - "
            f"Train Loss: {train_loss:.6f} - "
            f"LR: {self.scheduler.get_last_lr()[0]:e}"
        )
        self.print(summary)

    @torch.no_grad()
    def _log_preds(self, inputs: torch.Tensor):
        if self.rank == 0:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.ddp_model(inputs, is_causal=True)
            pred_token_ids = torch.argmax(logits, dim=-1)
            preds = self.train_dataset.tokenizer.decode(pred_token_ids[0])
            with self.path["predictions"].open("a") as preds_file:
                preds_file.write(
                    "############# "
                    f"Iteration {int(self.grad_steps)}"
                    " #############"
                    f"\n{preds}\n"
                )

    def _log_stats(self):
        self.train_loss_metric.log(int(self.grad_steps))
        if self.grad_steps % self.config["validate_every_n_steps"] == 0:
            self.val_loss_metric.log(int(self.grad_steps))

    def train(self):
        self.start_train_time = self.start_grad_update_time = perf_counter()
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

                    self._summary()
                
                if self.grad_steps % self.config["reset_metrics_every_n_steps"] == 0:
                    self.train_loss_metric.reset()

                if self.grad_steps % self.config["log_preds_every_n_steps"] == 0:
                    self._log_preds(inputs)

                if self.grad_steps % self.config["save_every_n_steps"] == 0:
                    self.print("Saving checkpoint...")
                    self.save_ckpt()

                if self.grad_steps % self.config["validate_every_n_steps"] == 0:
                    self.print("Validating...")
                    start_val_time = perf_counter()
                    self.validate()
                    val_time = str(timedelta(seconds=perf_counter() - start_val_time))[
                        :-3
                    ]
                    val_loss = self.val_loss_metric.compute()
                    self.print(f"Val Time: {val_time} - Val Loss: {val_loss}")

                if self.iterations % self.config["grad_accumulation_steps"] == 0:
                    self._log_stats()
                    self.start_grad_update_time = perf_counter()


    @torch.no_grad()
    def validate(self) -> float:
        self.val_loss_metric.reset()
        for token_ids in self.val_dataset:
            token_ids = token_ids.to(self.rank)
            inputs, targets = token_ids[:, :-1], token_ids[:, 1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.ddp_model(inputs, is_causal=True)
                loss = self.ddp_model.module.loss(
                    logits, targets
                )

            self.val_loss_metric.update(loss.item())
