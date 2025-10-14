import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import BinaryIO

import regex as re
from tqdm import tqdm

from cs336_basics.consts import MAX_CHUNK_SIZE, NUM_PROCESSES, PAT


def find_chunk_boundaries(
    file: BinaryIO,
    num_processes: int,
    max_chunk_size: int,
    pattern: re.Pattern[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = min(file_size // num_processes, max_chunk_size)
    desired_num_chunks = file_size // chunk_size

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the first special token in the mini chunk
            match = pattern.search(mini_chunk)
            found_at = match.start() if match else -1

            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    pattern: re.Pattern[str],
) -> Counter[str]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunks = pattern.split(chunk)
    counter = Counter()
    for ch in chunks:
        counter.update(Counter(m.group(0) for m in PAT.finditer(ch)))
    return counter


def pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int = NUM_PROCESSES,
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> Counter[str]:
    byte_pattern = re.compile(
        b"|".join(sorted([re.escape(token.encode("utf-8")) for token in special_tokens], key=len, reverse=True))
    )

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, max_chunk_size, byte_pattern)

    # Pattern for matching special tokens
    pattern = re.compile("|".join(sorted(map(re.escape, special_tokens), key=len, reverse=True)))

    with ProcessPoolExecutor(max_workers=num_processes) as ex:
        # ex.map gives you an iterator over results in order
        it = ex.map(process_chunk, repeat(input_path), boundaries[:-1], boundaries[1:], repeat(pattern))

        return sum(tqdm(it, total=len(boundaries) - 1), Counter())


if __name__ == "__main__":
    import pickle

    from cs336_basics.consts import SPECIAL_TOKENS

    # INPUT_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
    INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
    OUTPUT_PATH = "data/pretokens.pkl"

    counter = pretokenize(INPUT_PATH, SPECIAL_TOKENS)
    print("Number of pre-tokens:", len(counter))

    print(f"Saving to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(counter, f)
