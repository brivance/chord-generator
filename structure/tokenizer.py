import torch
import json

from utils import START_TOKEN, END_TOKEN, PADDING_TOKEN, MAX_SEQ_LEN


def save_tokenizer(tokenizer, filepath):
    # Save the vocabulary and any other necessary attributes
    tokenizer_data = {
        "vocab": tokenizer.vocab,  # or however you access your vocabulary
    }
    with open(filepath, "w") as f:
        json.dump(tokenizer_data, f)


def load_tokenizer(filepath):
    with open(filepath, "r") as f:
        tokenizer_data = json.load(f)
    # Reconstruct the tokenizer object
    tokenizer = ChordTokenizer()  # instantiate your tokenizer class
    tokenizer.vocab = tokenizer_data["vocab"]
    return tokenizer


class ChordTokenizer:
    def __init__(self):
        self.vocab = [
            START_TOKEN,
            END_TOKEN,
            PADDING_TOKEN,
        ]  # this array is just for encoding/decoding ease

    def create_vocab(self, sequences):
        for sequence in sequences:
            for chord in sequence:
                if chord not in self.vocab:
                    self.vocab.append(chord)

    def tokenize(self, song, start_token=False, end_token=False):
        chord_indices = [self.vocab.index(token) for token in song]
        if start_token:
            chord_indices.insert(0, self.vocab.index(START_TOKEN))
        if end_token:
            chord_indices.append(self.vocab.index(END_TOKEN))
        for _ in range(len(chord_indices), MAX_SEQ_LEN):
            chord_indices.append(self.vocab.index(PADDING_TOKEN))
        return torch.tensor(chord_indices)

    def decode_sequence(self, token_ids):
        return [
            self.vocab[token_id] if token_id < len(self.vocab) else self.vocab[0]
            for token_id in token_ids
        ]

    def get_vocab_size(self):
        return len(self.vocab)

    def __getitem__(self, token):
        if token in self.vocab:
            return self.vocab.index(token)
        else:
            raise ValueError(f"Token '{token}' not found in the vocabulary.")
