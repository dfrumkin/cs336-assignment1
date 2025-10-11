import pickle
from collections.abc import Iterable, Iterator
from typing import Self


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
        # TODO: add special tokens to vocab.  What about our special tokens?
        self._vocab = vocab
        self._merges = merges

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
        # TODO
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.

        Args:
            iterable (Iterable[str]): An iterable of strings

        Yields:
            Iterator[int]: A generator yielding token IDs
        """
        # TODO
        pass

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Args:
            ids (list[int]): A sequence of token IDs

        Returns:
            str: Decoded text
        """
        # TODO
        pass
