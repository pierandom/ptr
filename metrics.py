from typing import Optional
import csv
from pathlib import Path
import torch
import torch.distributed as dist


class AverageMetric:
    def __init__(self, rank: int, world_size: int, log_path: Path):
        self.rank = rank
        self.world_size = world_size
        self.values = []
        if self.rank == 0:
            self.writer = csv.writer(log_path.open("a"))

    def update(self, v: float):
        self.values.append(v)

    def compute(self) -> Optional[float]:
        if not self.values:
            return None
        
        v: torch.Tensor = torch.Tensor(self.values).to(self.rank) / self.world_size
        dist.all_reduce(v)
        return v.mean().item()

    def reset(self):
        self.values = []
    
    def log(self, iteration: int):
        val = self.compute()
        if self.rank == 0:
            self.writer.writerow((iteration, val))
