import torch
import torch.distributed as dist


class AverageMetric:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.values = []

    def update(self, v: float):
        self.values.append(v)

    def compute(self) -> float:
        v: torch.Tensor = torch.Tensor(self.values).to(self.rank) / self.world_size
        dist.all_reduce(v)
        return v.mean().item()

    def reset(self):
        self.values = []
