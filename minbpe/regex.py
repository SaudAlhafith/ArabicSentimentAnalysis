import regex as re
from .base import Tokenizer, get_stats, merge, render_token
import multiprocessing as mp


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def _parallel_get_stats(chunk):
    # Compute stats for a single chunk (starting with an empty dictionary)
    return get_stats(chunk, {})

def _parallel_merge(chunk, pair, idx):
    # Replace occurrences of 'pair' in the chunk with the new token id 'idx'
    return merge(chunk, pair, idx)

class RegexTokenizer(Tokenizer):
    
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        # Create a pool using all available cores
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for i in range(num_merges):
                # Parallel counting: compute stats for each chunk
                partial_stats = pool.map(_parallel_get_stats, ids)
                stats = {}
                for d in partial_stats:
                    for k, v in d.items():
                        stats[k] = stats.get(k, 0) + v

                # Choose the pair with the maximum frequency
                pair = max(stats, key=stats.get)
                idx = 256 + i

                # Parallel merge: update each chunk with the merged pair
                ids = pool.starmap(_parallel_merge, [(chunk, pair, idx) for chunk in ids])
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

                if verbose:
                    print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):

        # convert text bytes to int list
        ids = list(text_bytes)

        # while there is bytes more than 2 "we can merge"
        while len(ids) >= 2:

            # count pairs, and choose the first merged pair, self.merges.get(p) gives us the index of the merge
            stats = get_stats(ids)
            pair = min(stats, key= lambda p: self.merges.get(p, float("inf")))

            # if it returned infinity then it didn't find the minimum among the merges then it will return the first element 
            if pair not in self.merges:
                break

            # get the idx for the pair merged (where idx -> p0, p1)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def encode_ordinary(self, text):

        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special="none_raise"):
        special = None
        if allowed_special == 'all':
            special = self.special_tokens
        elif allowed_special == 'none':
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")


        if not special:
            return self.encode_ordinary(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.append(self.encode_ordinary(part))

        return ids

        