import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

PAD, START, END, UNK, EN = range(5)


@dataclass(frozen=True)
class Dataset:
    train: TensorDataset
    evaluation: TensorDataset
    decode: Callable[[list[int]], str]


class TokenPairUnpickler(pickle.Unpickler):
    """pecSC token pairs contain only Python lists, tuples, and integers."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Objects are forbidden in token pairs: {module}.{name}")


def read_pairs(path: Path, width: int, vocab_size: int, allow_empty=False):
    if path.suffix == ".pkl":
        with path.open("rb") as stream:
            pairs = TokenPairUnpickler(stream).load()
    else:
        pairs = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for pair in pairs:
        if pair is None and allow_empty:
            rows.append((START, EN, END))
            continue
        source, target = pair
        if (
            tuple(source) != tuple(target)
            or not 4 <= len(source) <= width
            or tuple(source[:2]) != (START, EN)
            or source[-1] != END
            or any(type(token) is not int or token == PAD or not 0 <= token < vocab_size
                   for token in source)
            or any(token in {START, END, EN, 5, 6, 7} for token in source[2:-1])
        ):
            raise ValueError(f"Invalid English reconstruction pair in {path.name}")
        rows.append(tuple(source))
    return rows


def build_data(train, evaluation, width, decode):
    if len(train) < 2 or not evaluation:
        raise ValueError("Need at least two training records and one evaluation record")
    if not set(train).isdisjoint(evaluation):
        raise ValueError("Training and evaluation share an encoded sentence")

    def pad(rows):
        return TensorDataset(torch.tensor(
            [row + (PAD,) * (width - len(row)) for row in rows], dtype=torch.long
        ))

    return Dataset(pad(train), pad(evaluation), decode)


def text_decoder(decode):
    def decode_sentence(ids):
        pieces = []
        for token in ids:
            if token == END:
                break
            if token == UNK or token >= 8:
                pieces.append(token)
        return decode(pieces).strip()

    return decode_sentence


def load_data(train_path, evaluation_path, tokenizer_path, width, vocab_size):
    import sentencepiece

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    expected = ("<PAD>", "<START>", "<END>", "<UNK>", "<EN>", "<PT>", "<ES>", "<FR>")
    if tokenizer.vocab_size() != vocab_size or any(
        tokenizer.id_to_piece(index) != token for index, token in enumerate(expected)
    ):
        raise ValueError("Tokenizer does not match the pecSC BPE vocabulary")

    return build_data(
        read_pairs(train_path, width, vocab_size, allow_empty=True),
        read_pairs(evaluation_path, width, vocab_size),
        width,
        text_decoder(tokenizer.decode),
    )


def synthetic_data():
    words = ["send", "receive", "signal", "message", "noise", "channel", "text", "data"]
    rows = [(START, EN, first, second, END)
            for first in range(8, 16) for second in range(8, 16)]
    return build_data(
        rows[::2], rows[1::8], 6,
        text_decoder(lambda ids: " ".join(words[i - 8] if i >= 8 else "<unk>" for i in ids)),
    )
