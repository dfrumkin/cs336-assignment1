import os
import time
from collections import Counter
from multiprocessing import Process, Queue

from tqdm import tqdm

from cs336_basics.consts import NUM_PROCESSES
from cs336_basics.pretokenizer import pretokenize


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = NUM_PROCESSES,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    point1 = time.perf_counter()
    pretoken_counter = pretokenize(input_path, special_tokens, num_processes)
    point2 = time.perf_counter()
    print(f"Pretokenizer time: {point2 - point1:.3f}s")
    return _train_bpe_helper(pretoken_counter, vocab_size, special_tokens, num_processes)


def actor(in_q: Queue, out_q: Queue):
    pretokens = []
    new_token_ind = -1
    while True:
        msg = in_q.get()
        if msg["type"] == "bind":
            # Initialize pretokens and count pairs
            pretokens = msg["pretokens"]
            new_token_ind = msg["vocab_size"]
            shift = msg["shift"]

            pair_counter = Counter()
            for i, (pretoken, factor) in enumerate(pretokens):
                pretoken = tuple(c + shift for c in pretoken.encode("utf-8"))
                counter = Counter(pretoken[i : i + 2] for i in range(len(pretoken) - 1))
                counter = Counter({k: v * factor for k, v in counter.items()})
                pretokens[i] = (pretoken, factor)
                pair_counter.update(counter)

            out_q.put(pair_counter)
        elif msg["type"] == "round":
            # Perform a merge
            pair = msg["pair"]
            token_counter = Counter()

            for pt_ind, (pretoken, ptk_cnt) in enumerate(pretokens):
                new_pretoken = []
                i = 0
                while i < len(pretoken):
                    if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == pair:
                        # Update the previous token count
                        if i > 0:
                            prev_token_ind = new_pretoken[-1]
                            token_counter[(prev_token_ind, new_token_ind)] += ptk_cnt
                            token_counter[(prev_token_ind, pair[0])] -= ptk_cnt

                        # Update the next token count
                        if i < len(pretoken) - 2:
                            next_token_ind = pretoken[i + 2]
                            token_counter[(new_token_ind, next_token_ind)] += ptk_cnt
                            token_counter[(pair[1], next_token_ind)] -= ptk_cnt

                        new_pretoken.append(new_token_ind)
                        i += 2
                    else:
                        new_pretoken.append(pretoken[i])
                        i += 1

                # Update the pretoken
                pretokens[pt_ind] = (new_pretoken, ptk_cnt)

            # Output the counters
            out_q.put(token_counter)

            # Keep track of the new token id
            new_token_ind += 1
        elif msg["type"] == "stop":
            break


def _train_bpe_helper(
    pretoken_counter: Counter[str],
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = NUM_PROCESSES,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Initialize vocabulary and merges
    token_list = [token.encode("utf-8") for token in special_tokens] + [bytes([i]) for i in range(256)]
    vocab = dict(enumerate(token_list))
    merges = []
    num_iters = vocab_size - len(vocab)
    assert num_iters >= 0

    # Split data
    pretokens = list(pretoken_counter.items())
    num_pretokens = len(pretokens)
    chunk_size = (num_pretokens + num_processes - 1) // num_processes
    chunks = [pretokens[i : i + chunk_size] for i in range(0, num_pretokens, chunk_size)]
    del pretokens

    # Start processes
    in_queues, out_queues, procs = [], [], []
    for _ in range(num_processes):
        iq, oq = Queue(), Queue()
        p = Process(target=actor, args=(iq, oq), daemon=True)
        p.start()
        in_queues.append(iq)
        out_queues.append(oq)
        procs.append(p)

    # Bind inputs
    for iq, ch in zip(in_queues, chunks, strict=False):
        iq.put({"type": "bind", "pretokens": ch, "vocab_size": len(vocab), "shift": len(special_tokens)})

    # Initialize the counter
    pair_counter = sum((oq.get() for oq in out_queues), Counter())

    # Merge greedily
    for _ in tqdm(range(num_iters)):
        # Find the most frequent pair using the lexicographic order of the original byte sequences to resolve ties
        max_count = max(pair_counter.values())
        candidates = [p for p, c in pair_counter.items() if c == max_count]
        pair = max(candidates, key=lambda p: (vocab[p[0]], vocab[p[1]]))

        # Define the new token
        new_token_ind = len(vocab)

        # Update pair counter
        for iq in in_queues:
            iq.put({"type": "round", "pair": pair})
        for oq in out_queues:
            pair_counter.update(oq.get())

        # Update vocab, merges, and counts for the pair
        token1_bytes = vocab[pair[0]]
        token2_bytes = vocab[pair[1]]
        vocab[new_token_ind] = token1_bytes + token2_bytes
        merges.append((token1_bytes, token2_bytes))
        del pair_counter[pair]

    for iq in in_queues:
        iq.put({"type": "stop"})
    for p in procs:
        p.join()

    return vocab, merges


if __name__ == "__main__":
    import pickle
    import resource
    import sys

    from cs336_basics.consts import SPECIAL_TOKENS

    DEBUG = False

    if DEBUG:
        PRETOKEN_PATH = "data/pretokens.pkl"
        VOCAB_SIZE = 1000

        with open(PRETOKEN_PATH, "rb") as f:
            pretoken_counter = pickle.load(f)

        vocab, merges = _train_bpe_helper(pretoken_counter, VOCAB_SIZE, SPECIAL_TOKENS)
        print(f"BPE has finished: vocabulary of size {len(vocab)} and {len(merges)} merges")
    else:
        # scalene --html --cpu --memory cs336_basics/train_bpe.py
        TINY_STORIES = False

        if TINY_STORIES:
            INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
            VOCAB_OUTPUT_PATH = "data/tiny_train_vocab.pkl"
            MERGES_OUTPUT_PATH = "data/tiny_train_merges.pkl"
            VOCAB_SIZE = 10000
        else:
            INPUT_PATH = "data/owt_train.txt"
            VOCAB_OUTPUT_PATH = "data/owt_train_vocab.pkl"
            MERGES_OUTPUT_PATH = "data/owt_train_merges.pkl"
            VOCAB_SIZE = 32000

        start = time.perf_counter()
        vocab, merges = train_bpe(INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS)
        end = time.perf_counter()

        # This is an overestimation (sum of peak usages)
        peak_usage = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        )
        peak_memory_mb = peak_usage / (1024**2 if sys.platform == "darwin" else 1024)

        print(f"Total time: {end - start:.3f}s; memory usage: {peak_memory_mb:.1f} MB")

        # pickle is fine since dict, list, tuple, int, bytes - are primitive datatypes, stable across versions
        with open(VOCAB_OUTPUT_PATH, "wb") as f:
            pickle.dump(vocab, f)

        with open(MERGES_OUTPUT_PATH, "wb") as f:
            pickle.dump(merges, f)
