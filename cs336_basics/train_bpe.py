import os
from collections import Counter
from cs336_basics.pretokenizer import pretokenize


def tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretoken_counter = pretokenize(input_path, special_tokens)
    return _tokenizer_inner(pretoken_counter, vocab_size, special_tokens)


def _tokenizer_inner(
        pretoken_counter: Counter, 
        vocab_size: int, 
        special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # Initialize vocabulary and merges
    token_list = [token.encode("utf-8") for token in special_tokens] + [bytes([i]) for i in range(256)]
    vocab = dict(enumerate(token_list))
    merges = []
    num_iters = vocab_size - len(vocab)
    assert num_iters >= 0

    # Prepare pretokens using the vocabulary
    shift = len(special_tokens)
    pretokens = [[tuple(c + shift for c in seq.encode("utf-8")), cnt] for seq, cnt in pretoken_counter.items()]

    # Count pairs
    pair_counter = Counter()
    
    for pretoken, factor in pretokens:
        counter = Counter(pretoken[i: i+2] for i in range(len(pretoken) - 1))
        counter = Counter({k: v * factor for k, v in counter.items()})
        pair_counter.update(counter)
    
    # Merge greedily
    for _ in range(num_iters):
        # Find the most frequent pair using the lexicographic order of the original byte sequences to resolve ties
        max_count = max(pair_counter.values())
        candidates = [p for p, c in pair_counter.items() if c == max_count]
        pair = max(candidates, key=lambda p: (vocab[p[0]], vocab[p[1]]))
        
        # Update pretokens
        new_token_ind = len(vocab)

        for pt_ind, (pretoken, ptk_cnt) in enumerate(pretokens):
            new_pretoken = []
            i = 0
            while i < len(pretoken):
                if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == pair:
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
            pretokens[pt_ind][0] = new_pretoken

        # Update vocab, merges, and counts for the pair
        token1_bytes = vocab[pair[0]]
        token2_bytes = vocab[pair[1]]
        vocab[new_token_ind] = token1_bytes + token2_bytes
        merges.append((token1_bytes, token2_bytes))
        del pair_counter[pair]

    return vocab, merges


if __name__ == "__main__":
    import pickle

    SPECIAL_TOKENS = ["<|endoftext|>"]
    DEBUG = False

    if DEBUG:
        PRETOKEN_PATH = "data/pretokens.pkl"
        VOCAB_SIZE = 1000

        with open(PRETOKEN_PATH, "rb") as f:
            pretoken_counter = pickle.load(f)

        vocab, merges = _tokenizer_inner(pretoken_counter, VOCAB_SIZE, SPECIAL_TOKENS)
        print(f"BPE has finished: vocabulary of size {len(vocab)} and {len(merges)} merges")
    else:
        # scalene --html --reduced-profile --cpu --memory cs336_basics/train_bpe.py
        TINY_STORIES = False

        if TINY_STORIES:
            INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
            OUTPUT_PATH = "data/tiny_train.pkl"
            VOCAB_SIZE = 10000
        else:
            INPUT_PATH = "data/owt_train.txt"
            OUTPUT_PATH = "data/owt_train.pkl"
            VOCAB_SIZE = 32000

        vocab, merges = tokenizer(INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS)

        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump((vocab, merges), f)
