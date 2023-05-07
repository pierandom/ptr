import json

from tokenizers import (
    models,
    pre_tokenizers,
    trainers,
    decoders,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

def get_training_corpus(buffer_size=1024):
    wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train").to_iterable_dataset()
    buffer = []
    for example in wiki_dataset:
        buffer.append(example["text"])
        if len(buffer) == buffer_size:
            yield buffer
            buffer = []
    yield buffer


def build_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=2**15+2**14,
        special_tokens=["<|end|>"]
    )
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="<|end|>"
    )
    return wrapped_tokenizer


if __name__ == "__main__":
    tokenizer = build_tokenizer()
    tokenizer.save_pretrained("tokenizer_ptr")
