import json
import random
import itertools
import torch
from torch.utils.data import IterableDataset


class NextTokenDataset(IterableDataset):
    """Dataset for Next Token prediction task (aka Causal Language Modeling).

    Implements Python's iterator protocol for Data Parallel training.
    The dataset pre-fetches a bunch of examples, then fills and shuffles a buffer
    and returns a micro-batch to a given process without overlap.
    """

    def __init__(
        self,
        split: str,
        rank: int,
        world_size: int,
        tokenizer,
        batch_size: int,
        context_len: int,
        buffer_factor: int = 100,
        seed: int = 42,
    ):
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.buffer_size = batch_size * buffer_factor
        self.context_len = context_len
        self._state = dict(shard_id=0, processed_examples=0, seed=seed)

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self._state = state

    def _train_iter(self):
        start_shard_id = self._state["shard_id"]
        iterator = open(f"/mnt/reginald/thepile/train/{start_shard_id:02}.jsonl")
        itertools.islice(iterator, self._state["processed_examples"], None)
        for shard_id in range(start_shard_id, 30):
            for example in iterator:
                self._state["processed_examples"] += 1
                yield example
            iterator = open(f"/mnt/reginald/thepile/train/{shard_id:o2}.jsonl")
            self._state["processed_examples"] = 0
            self._state["shard_id"] = shard_id
        self._state["shard_id"] = 0
        self._state["seed"] += 1

    def _iter(self):
        if self.split == "train":
            return self._train_iter()
        else:
            return open(f"/mnt/reginald/thepile/{self.split}.jsonl")

    def _example(self):
        token_ids = []
        for item_line in self._iter():
            item = json.loads(item_line)
            token_ids.extend(
                self.tokenizer.encode(item["text"] + self.tokenizer.eos_token)
            )
            while len(token_ids) >= self.context_len:
                yield torch.LongTensor(token_ids[: self.context_len])
                token_ids = token_ids[self.context_len :]

    def _buffer(self):
        buffer = []
        for example in self._example():
            buffer.append(example)
            if len(buffer) == self.buffer_size:
                random.shuffle(buffer)
                yield buffer
                buffer = []

    def __iter__(self):
        random.seed(self._state["seed"])
        for buffer in self._buffer():
            while buffer:
                yield torch.stack(buffer[self.rank : self.batch_size : self.world_size])
                buffer = buffer[self.batch_size :]
