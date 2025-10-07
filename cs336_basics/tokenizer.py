import os
from collections import Counter
from .pretokenizer import pretokenize

NUM_PROCESSES = min(32, os.cpu_count())


def tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    pretoken_counter = pretokenize(input_path, NUM_PROCESSES, special_tokens)
    return _tokenizer_inner(pretoken_counter, vocab_size, special_tokens)


def _tokenizer_inner(
        pretoken_counter: dict[tuple[int], int], 
        vocab_size: int, 
        special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # Initialize vocabulary and merges
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)} | {i + len(special_tokens): bytes([i]) for i in range(256)}
    merges = []
    num_iters = vocab_size - len(vocab)
    assert num_iters >= 0

    # Count pairs
    pair_counter = Counter()
    pretokens = list(pretoken_counter.items())
    
    for pretoken, factor in pretokens:
        counter = Counter(pretoken[i: i+2] for i in range(len(pretoken) - 1))
        counter = Counter({k: v * factor for k, v in counter.items()})
        pair_counter.update(counter)
    
    for _ in range(num_iters):
        # Find the most frequent pair
        pair = max(pair_counter.items(), key=lambda kv: (kv[1], kv[0]))[0]
        new_token_ind = len(vocab)
        
        # Update pretokens
        for pt_ind, (pretoken, ptk_cnt) in enumerate(pretokens):
            new_pretoken = []
            i = 0
            while i < len(pretoken) - 1:
                if (pretoken[i], pretoken[i + 1]) == pair:
                    # Update the previous pair
                    if i > 0:
                        last_token = new_pretoken[-1]
                        pair_counter[(last_token, pretoken[i])] -= ptk_cnt
                        pair_counter[(last_token, new_token_ind)] += ptk_cnt

                    # Update the next pair
                    if i < len(pretoken) - 2:
                        pair_counter[(pretoken[i + 1], pretoken[i + 2])] -= ptk_cnt
                        pair_counter[(new_token_ind, pretoken[i + 2])] += ptk_cnt

                    new_pretoken.append(new_token_ind)
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1

            # Update the pretoken
            pretokens[pt_ind] = (new_pretoken, ptk_cnt)

        # Update vocab, merges, and counts for the pair
        token1_bytes = vocab[pair[0]]
        token2_bytes = vocab[pair[1]]
        vocab[new_token_ind] = token1_bytes + token2_bytes
        merges.append((token1_bytes, token2_bytes))
        del pair_counter[pair]

    return vocab, merges


if __name__ == "__main__":
    import pickle

    PRETOKEN_PATH = "data/pretokens.pkl"
    SPECIAL_TOKEN = "<|endoftext|>"
    VOCAB_SIZE = 1000

    with open(PRETOKEN_PATH, "rb") as f:
        pretoken_counter = pickle.load(f)

    vocab, merges = _tokenizer_inner(pretoken_counter, VOCAB_SIZE, [SPECIAL_TOKEN])
    print(f"BPE has finished: vocabulary of size {len(vocab)} and {len(merges)} merges")
