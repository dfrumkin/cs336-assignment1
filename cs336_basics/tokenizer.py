import heapq
import pickle
from collections.abc import Iterable, Iterator
from typing import Self

import regex as re

from cs336_basics.consts import PAT, SPECIAL_TOKENS


class PairHeap:
    """Min-heap of candidate merges with staleness checks via version counters."""

    def __init__(self, ids: list[int], merges: dict[tuple[int, int], int]) -> None:
        self._ids = ids
        self._merges = merges
        self._pair_rank = {pair: rank for rank, pair in enumerate(merges)}
        n_tokens = len(ids)

        # Build initial entries only for adjacent, mergeable pairs
        items = []
        for i in range(n_tokens - 1):
            rank = self._pair_rank.get((ids[i], ids[i + 1]))
            if rank is not None:
                items.append((rank, i, i + 1, 0, 0))  # (rank, i, j, ver_i, ver_j)

        self._heap = items
        heapq.heapify(self._heap)

        # Build a linked list with versioning for fast removal
        self._prev = list(range(-1, n_tokens - 1))
        self._next = list(range(1, n_tokens)) + [-1]
        self._ver = [0] * n_tokens

    def _push(self, i: int, j: int) -> None:
        rank = self._pair_rank.get((self._ids[i], self._ids[j]))
        if rank is not None:
            heapq.heappush(self._heap, (rank, i, j, self._ver[i], self._ver[j]))

    def _pop(self) -> tuple[int, int] | None:
        pair = None
        while self._heap:
            _, i, j, vi, vj = heapq.heappop(self._heap)
            if self._ver[i] == vi and self._ver[j] == vj:
                pair = (i, j)
                break
        return pair

    def merge(self) -> list[int]:
        while True:
            top = self._pop()
            if not top:
                break

            i, j = top

            # Put the merged token at i
            self._ids[i] = self._merges[(self._ids[i], self._ids[j])]

            # Remove j; relink i->ri
            ri = self._next[j]
            self._next[i] = ri
            if ri != -1:
                self._prev[ri] = i

            # Update versions
            self._ver[i] += 1
            self._ver[j] = -1  # no longer used

            # Push only affected neighbors
            li = self._prev[i]
            if li != -1:
                self._push(li, i)
            if ri != -1:
                self._push(i, ri)

        # Collect surviving ids from the linked list
        out: list[int] = []
        cur = 0
        while self._ver[cur] == -1 and cur != -1:
            cur = self._next[cur]
        while cur != -1:
            out.append(self._ids[cur])
            cur = self._next[cur]
        return out


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        """Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        Args:
            vocab (dict[int, bytes]): Vocabulary
            merges (list[tuple[bytes, bytes]]): Ordered byte sequence merges
            special_tokens (list[str] | None, optional): Additional special tokens (optional)
        """
        if special_tokens is None:
            special_tokens = []

        # Pattern for matching special tokens
        all_special_tokens = set(SPECIAL_TOKENS + special_tokens)
        self._pat_special = re.compile(
            "(" + "|".join(map(re.escape, sorted(all_special_tokens, key=len, reverse=True))) + ")"
        )

        # Add special tokens to the vocabulary if they are not already there
        new_tokens = list(set(special_tokens) - set(v.decode("utf-8", errors="replace") for v in vocab.values()))
        self._vocab = vocab | dict(enumerate((token.encode("utf-8") for token in new_tokens), len(vocab)))
        self._inverse_vocab = {v: k for k, v in self._vocab.items()}

        # Prepare merges data in token indices form
        self._merges = {
            (self._inverse_vocab[b1], self._inverse_vocab[b2]): self._inverse_vocab[b1 + b2] for (b1, b2) in merges
        }

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Self:
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
            (in the same format that your BPE training code output) and (optionally) a list of special tokens.

        Args:
            vocab_filepath (str): Path to a serialized vocabulary
            merges_filepath (str): Path to serialized merges
            special_tokens (list[str] | None, optional): Additional special tokens (optional)

        Returns:
            Tokenizer: The constructed tokenizer
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs.

        Args:
            text (str): Input text

        Returns:
            list[int]: Sequence of token IDs
        """
        segs = self._pat_special.split(text)
        out: list[int] = []
        for idx, seg in enumerate(segs):
            # text, special, text, special, ... - and text may be emtpy
            if seg:
                if idx % 2 == 0:
                    for m in PAT.finditer(seg):
                        pretoken = seg[m.start() : m.end()]
                        tokens = [self._inverse_vocab[bytes([b])] for b in pretoken.encode("utf-8")]
                        pair_heap = PairHeap(tokens, self._merges)
                        out.extend(pair_heap.merge())
                else:
                    out.append(self._inverse_vocab[seg.encode("utf-8")])
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.

        Args:
            iterable (Iterable[str]): An iterable of strings

        Yields:
            Iterator[int]: A generator yielding token IDs
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Args:
            ids (list[int]): A sequence of token IDs

        Returns:
            str: Decoded text
        """
        try:
            data = b"".join(self._vocab[i] for i in ids)
        except KeyError as e:
            raise KeyError(f"Unknown token id: {e.args[0]}") from None
        return data.decode("utf-8", errors="replace")
