import os

import regex as re

SPECIAL_TOKENS = ["<|endoftext|>"]
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
NUM_PROCESSES = min(32, os.cpu_count() or 1)
MAX_CHUNK_SIZE = 2**25
