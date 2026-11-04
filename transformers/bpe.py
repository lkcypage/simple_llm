
import json
from collections import Counter


class BPE:
    def __init__(self, vocab_size=20000000):
        self.vocab_size = vocab_size

        # Os primeiros 256 tokens representam bytes UTF-8
        self.merges = []
        self.merge_ranks = {}

        self.special_tokens = {
            "<|pad|>": 0,
            "<|bos|>": 1,
            "<|eos|>": 2,
            "<|unk|>": 3
        }

        self.special_ids = {
            v: k for k, v in self.special_tokens.items()
        }

        self.base_vocab_size = 256

    @property
    def vocab_size_actual(self):
        return (
            self.base_vocab_size
            + len(self.merges)
            + len(self.special_tokens)
        )

    @property
    def vocab_size_model(self):
        return self.vocab_size_actual

    def train(self, texts, min_frequency=2):

        sequences = [
            list(text.encode("utf-8"))
            for text in texts
            if text
        ]

        self.merges = []
        self.merge_ranks = {}

        num_merges = max(0, self.vocab_size - 256)

        for step in range(num_merges):

            pair_counts = Counter()

            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1

            if not pair_counts:
                break

            best_pair, frequency = pair_counts.most_common(1)[0]

            if frequency < min_frequency:
                break

            new_id = 256 + len(self.merges)

            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = new_id

            new_sequences = []

            for seq in sequences:

                merged = []
                i = 0

                while i < len(seq):

                    if (
                        i < len(seq) - 1
                        and (seq[i], seq[i + 1]) == best_pair
                    ):
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(seq[i])
                        i += 1

                new_sequences.append(merged)

            sequences = new_sequences

            if (step + 1) % 100 == 0:
                print(
                    f"BPE: {step + 1}/{num_merges} merges"
                )

        print("Treinamento BPE concluído!")
        print("Merges aprendidos:", len(self.merges))
        print("Vocabulário:", self.vocab_size_actual)

    def encode(self, text, add_special_tokens=False):

        tokens = list(text.encode("utf-8"))

        while len(tokens) > 1:

            best_rank = None
            best_position = None

            for i in range(len(tokens) - 1):

                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)

                if rank is not None:
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_position = i

            if best_position is None:
                break

            i = best_position

            tokens = (
                tokens[:i]
                + [best_rank]
                + tokens[i + 2:]
            )

        if add_special_tokens:
            tokens = [
                self.special_tokens["<|bos|>"]
            ] + tokens + [
                self.special_tokens["<|eos|>"]
            ]

        return tokens

    def decode(self, tokens, skip_special_tokens=True):

        byte_sequence = bytearray()

        for token in tokens:

            if token in self.special_ids:

                if skip_special_tokens:
                    continue

                byte_sequence.extend(
                    self.special_ids[token].encode("utf-8")
                )

            elif token < 256:

                byte_sequence.append(token)

            else:

                merge_index = token - 256

                if merge_index >= len(self.merges):
                    raise ValueError(f"Token inválido: {token}")

                # Reconstrói os bytes recursivamente
                byte_sequence.extend(
                    self._decode_merge(token)
                )

        return byte_sequence.decode(
            "utf-8",
            errors="replace"
        )

    def _decode_merge(self, token):

        if token < 256:
            return bytes([token])

        left, right = self.merges[token - 256]

        return (
            self._decode_merge(left)
            + self._decode_merge(right)
        )

    def save(self, filename="bpe.json"):

        data = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "special_tokens": self.special_tokens
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        print("BPE salvo:", filename)

    def load(self, filename="bpe.json"):

        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.merges = [
            tuple(pair) for pair in data["merges"]
        ]

        self.special_tokens = data["special_tokens"]

        self.special_ids = {
            v: k for k, v in self.special_tokens.items()
        }

        self.merge_ranks = {
            pair: 256 + i
            for i, pair in enumerate(self.merges)
        }

        print("BPE carregado!")
