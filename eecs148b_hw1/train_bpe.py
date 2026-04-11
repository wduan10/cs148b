import os
import regex as re


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # split into chunks
    if special_tokens:
        pattern = "|".join(map(re.escape, special_tokens))
        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    # for each chunk, pretokenize with regex
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pretoken_counts = {}
    pretoken_splits = {}
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            pretoken = match.group(0).encode("utf-8")
            pretoken_counts[pretoken] = pretoken_counts.get(pretoken, 0) + 1

    for pretoken in pretoken_counts:
        pretoken_splits[pretoken] = [bytes([b]) for b in pretoken]

    # initialize vocab
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    for special_token in special_tokens:
        vocab[len(vocab)] = bytes(special_token, "utf-8")

    # count pairs of tokens
    pair_counts = {}
    pair_to_words = {}
    for pretoken, count in pretoken_counts.items():
        tokens = pretoken_splits[pretoken]
        for i in range(len(tokens) - 1):
            token1 = tokens[i]
            token2 = tokens[i + 1]
            pair_counts[(token1, token2)] = pair_counts.get(
                (token1, token2), 0) + count
            if (token1, token2) not in pair_to_words:
                pair_to_words[(token1, token2)] = set()
            pair_to_words[(token1, token2)].add(pretoken)

    merges = []
    while len(vocab) < vocab_size:
        # find most frequent pair of tokens
        best_pair, best_count = max(
            pair_counts.items(),
            key=lambda x: (x[1], x[0])
        )

        # update our dictionaries
        # concatenate the two bytes
        merged_bytes = best_pair[0] + best_pair[1]

        pair_counts[best_pair] = 0
        words_affected = pair_to_words[best_pair]
        for pretoken in words_affected:
            # update pair_counts, pretoken_splits, pair_to_words
            tokens = pretoken_splits[pretoken]
            new_tokens = []

            i = 0
            while i < len(tokens):
                # check if we can merge
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(merged_bytes)
                    if i > 0:
                        pair_counts[(tokens[i-1], tokens[i])] = pair_counts.get(
                            (tokens[i-1], tokens[i]), 0) - pretoken_counts[pretoken]

                        new_pair = (tokens[i-1], merged_bytes)
                        pair_counts[new_pair] = pair_counts.get(
                            new_pair, 0) + pretoken_counts[pretoken]
                        if new_pair not in pair_to_words:
                            pair_to_words[new_pair] = set()
                        pair_to_words[new_pair].add(pretoken)
                    if i < len(tokens) - 2:
                        pair_counts[(tokens[i+1], tokens[i+2])] = pair_counts.get(
                            (tokens[i+1], tokens[i+2]), 0) - pretoken_counts[pretoken]
                        
                        new_pair = (merged_bytes, tokens[i+2])
                        pair_counts[new_pair] = pair_counts.get(
                            new_pair, 0) + pretoken_counts[pretoken]
                        if new_pair not in pair_to_words:
                            pair_to_words[new_pair] = set()
                        pair_to_words[new_pair].add(pretoken)
                    i += 2  # skip both
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            pretoken_splits[pretoken] = new_tokens

        # merge pair of tokens and update pretoken_splits
        vocab[len(vocab)] = merged_bytes
        merges.append(best_pair)

    return vocab, merges
