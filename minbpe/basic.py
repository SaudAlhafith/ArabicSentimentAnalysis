from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        text_bytes = b"".join([self.vocab[idx] for idx in ids])
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):

        # convert text to bytes
        text_bytes = text.encode("utf-8")
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
        