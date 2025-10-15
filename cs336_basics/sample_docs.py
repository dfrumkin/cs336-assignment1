import random
from collections.abc import Iterator

END_OF_TEXT = b"<|endoftext|>"
CHUNK_SIZE = 2**28
SAMPLE_SIZE = 10


def stream_docs(path: str) -> Iterator[bytes]:
    """Yield documents separated by `delim` from a (potentially huge) file."""
    with open(path, "rb") as f:
        buf = b""
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            buf += chunk
            # Split by delimiter; keep the last piece in `buf` in case it's partial
            *docs, buf = buf.split(END_OF_TEXT)
            yield from docs
        # Emit the final trailing doc (if file doesn't end with the delimiter)
        if buf:
            yield buf


def sample_docs(path: str) -> list[bytes]:
    """Sample documents in a single pass using reservoir sampling."""
    reservoir: list[bytes] = []
    for i, doc in enumerate(stream_docs(path)):
        if i < SAMPLE_SIZE:
            reservoir.append(doc)
        else:
            j = random.randint(0, i - 1)  # inclusive
            if j < SAMPLE_SIZE:
                reservoir[j] = doc
    return reservoir


def create_sample(in_path: str, out_path: str) -> None:
    with open(out_path, "wb") as f:
        f.write(END_OF_TEXT.join(sample_docs(in_path)))


if __name__ == "__main__":
    random.seed(42)

    # Using validation data sets for speed
    TINY_IN = "data/TinyStoriesV2-GPT4-valid.txt"
    TINY_OUT = "data/TinyStoriesV2-GPT4-valid_sample.txt"
    OWT_IN = "data/owt_valid.txt"
    OWT_OUT = "data/owt_valid_sample.txt"

    create_sample(TINY_IN, TINY_OUT)
    create_sample(OWT_IN, OWT_OUT)
