class ChordDataset:
    def __init__(self, tokenized_input_data, tokenized_output_data, orig_chord_seqs):
        self.tokenized_input_data = tokenized_input_data
        self.tokenized_output_data = tokenized_output_data
        self.orig_chord_seqs = orig_chord_seqs

    def __len__(self):
        return len(self.tokenized_input_data)

    def get_chord_sequence(self, idx):
        return self.orig_chord_seqs[idx]

    def __getitem__(self, idx):
        return self.tokenized_input_data[idx], self.tokenized_output_data[idx]
