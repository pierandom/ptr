import random
import torch
from torch.utils.data import IterableDataset


class TinyStoriesDataset(IterableDataset):
    def __init__(
        self,
        split: str,
        rank: int,
        world_size: int,
        context_len: int,
        batch_size: int,
        tokenizer,
        shuffle: bool = False,
    ):
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.context_len = context_len
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.state = {}

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state

    def examples(self):
        token_ids = []
        with open(f"/mnt/reginald/tiny-stories/TinyStoriesV2-GPT4-{self.split}.txt") as file:
            for line in file:
                if "<|endoftext|>" in line:
                    token_ids.append(self.tokenizer.eos_token_id)
                    continue

                line = line.strip(" \n")
                token_ids.extend(self.tokenizer.encode(line + "\n"))

                while len(token_ids) >= self.context_len:
                    yield torch.LongTensor(token_ids[: self.context_len])
                    token_ids = token_ids[self.context_len :]

    def buffer(self):
        buffer = []
        for example in self.examples():
            buffer.append(example)
            if len(buffer) == self.batch_size:
                if self.shuffle:
                    random.shuffle(buffer)
                yield buffer
                buffer = []

    def __iter__(self):
        for buffer in self.buffer():
            while buffer:
                yield torch.stack(buffer[self.rank : self.batch_size : self.world_size])
                buffer = buffer[self.batch_size :]
