from argparse import ArgumentParser
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
    wiki_dataset = load_dataset(
        "wikipedia", "20220301.en", split="train"
    ).to_iterable_dataset()
    buffer = []
    for example in wiki_dataset:
        buffer.append(example["text"])
        if len(buffer) == buffer_size:
            yield buffer
            buffer = []
    yield buffer


def build_tokenizer(vocab_size: int) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["<|end|>"]
    )
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="<|end|>"
    )
    return wrapped_tokenizer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("tokenizer_name", type=str)
    parser.add_argument("--vocab_size", type=int, default=2**15)
    args = parser.parse_args()

    tokenizer = build_tokenizer(args.vocab_size)
    tokenizer.push_to_hub(args.tokenizer_name, commit_message="Add tokenizer")
    tokenizer.save_pretrained(args.tokenizer_name)
