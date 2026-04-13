import pickle
import regex as re
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        if special_tokens is not None:
            for special_token in special_tokens:
                token_bytes = special_token.encode("utf-8")
                if token_bytes not in vocab.values():
                    vocab[len(vocab)] = token_bytes

        self.vocab = vocab
        self.vocab_swapped = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = pickle.load(open(vocab_filepath, 'rb'))
        merges = pickle.load(open(merges_filepath, 'rb'))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(tok) for tok in specials) + ")"
            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        # for each chunk, pretokenize with regex
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        output = []
        for chunk in chunks:

            if self.special_tokens is not None and chunk in self.special_tokens:
                output.append(self.vocab_swapped[chunk.encode("utf-8")])
                continue
            
            pretokens = []
            for match in re.finditer(PAT, chunk):
                pretoken = match.group(0).encode("utf-8")
                pretoken_byte_array = [bytes([b]) for b in pretoken]
                pretokens.append(pretoken_byte_array)
            
            for pretoken in pretokens:
                for token1, token2 in self.merges:
                    new_pretoken = []
                    i = 0
                    while i < len(pretoken):
                        if i < len(pretoken) - 1 and pretoken[i] == token1 and pretoken[i + 1] == token2:
                            new_pretoken.append(token1 + token2)
                            i += 2
                        else:
                            new_pretoken.append(pretoken[i])
                            i += 1
                    pretoken = new_pretoken
                
                for token in pretoken:
                    output.append(self.vocab_swapped[token])
            
        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            ids = self.encode(chunk)
            for token_id in ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        bytes = [self.vocab[id] for id in ids]
        all_bytes = b"".join(bytes)
        return all_bytes.decode("utf-8", errors="replace")