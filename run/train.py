# pylint: disable=C0103
import json
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# from torch.nn import Transformer
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import Counter

import sys
import os

# Get the parent directory of the current file's directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from transformers import AdamW
from transformers import get_scheduler, TrainingArguments, Trainer
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
import numpy as np

from utils import exclude_chords
from structure.dataset import ChordDataset
from structure.tokenizer import ChordTokenizer, save_tokenizer

from utils import (
    create_masks,
    MAX_SEQ_LEN,
    PADDING_TOKEN,
    NEG_INF,
    BATCH_SIZE,
    d_model,
    dim_feedforward,
    num_heads,
    dropout,
    num_layers,
    LR,
    NUM_EPOCHS,
    EVAL_INTERVAL,
    device,
)

from structure.model_architecture import Transformer

# open the json data file and read in the chords as a 2d array
with open("data/song_chords.json", "r") as f:
    songs_data = json.load(f)

chord_sequences = []
for song in songs_data:
    if any(chord in exclude_chords for chord in song["chords"]):
        continue
    if len(song["chords"]) <= MAX_SEQ_LEN - 2:  # -2 for start and end tokens
        processed_chords = []
        for chord in song["chords"]:
            if "/" in chord:
                chord = chord.split("/")[0]
            chord = chord.replace("-", "b")  # Replace '-' with 'b'
            chord = chord.replace("(", "")  # Remove '('
            chord = chord.replace(")", "")  # Remove ')'
            chord = chord.replace("+", "aug")
            processed_chords.append(chord)
        chord_sequences.append(processed_chords)

all_chords = [chord for sequence in chord_sequences for chord in sequence]

# Count occurrences of each chord
chord_counts = Counter(all_chords)

# Print or inspect the chord counts
# print(chord_counts)

# print(f"Dataset number: {len(songs_data)}")
# print(f"Final number of songs: {len(chord_sequences)}")

# format the dataset with input and target sequences, create vocab, and tokenize
tokenizer = ChordTokenizer()
tokenizer.create_vocab(chord_sequences)

input_tokenized_dataset = []
output_tokenized_dataset = []
for chords in chord_sequences:
    input_tokenized_dataset.append(
        tokenizer.tokenize(chords, start_token=False, end_token=False)
    )
    output_tokenized_dataset.append(
        tokenizer.tokenize(chords, start_token=True, end_token=True)
    )

# print(tokenizer.vocab)

# print(len(tokenizer.vocab))

# print(input_tokenized_dataset[0])
# print(output_tokenized_dataset[0])

input_data_train, input_data_val = train_test_split(
    input_tokenized_dataset, test_size=0.2, random_state=42
)
output_data_train, output_data_val = train_test_split(
    output_tokenized_dataset, test_size=0.2, random_state=42
)

train_dataset = ChordDataset(input_data_train, output_data_train, chord_sequences)
val_dataset = ChordDataset(input_data_val, output_data_val, chord_sequences)


train_loader = DataLoader(train_dataset, BATCH_SIZE)
val_loader = DataLoader(val_dataset, BATCH_SIZE)
iterator = iter(train_loader)

# for batch_num, batch in enumerate(iterator):
#     print(batch)
#     if batch_num > 1:
#         break

model = Transformer(
    d_model,
    dim_feedforward,
    num_heads,
    dropout,
    num_layers,
    MAX_SEQ_LEN,
    tokenizer.get_vocab_size(),
    tokenizer.vocab,
)

criterian = nn.CrossEntropyLoss(
    ignore_index=tokenizer.vocab.index(PADDING_TOKEN), reduction="none"
)

# when computing the loss, we are ignoring cases when the label is the padding token
for params in model.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(model.parameters(), lr=LR)  # .00001 v .00005

model.train()
model.to(device)
total_loss = 0
graph_train_losses = []
graph_val_losses = []
for epoch in tqdm(range(NUM_EPOCHS)):
    epoch_training_loss = 0
    # print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        model.train()
        input_batch, output_batch = batch
        input_batch, output_batch = input_batch.to(device), output_batch.to(device)
        (
            encoder_self_attention_mask,
            decoder_self_attention_mask,
            decoder_cross_attention_mask,
        ) = create_masks(input_batch, output_batch, tokenizer)
        optim.zero_grad()
        train_predictions = model(
            input_batch,
            output_batch,
            encoder_self_attention_mask.to(device),
            decoder_self_attention_mask.to(device),
            decoder_cross_attention_mask.to(device),
        )
        labels = output_batch
        loss = criterian(
            train_predictions.view(-1, tokenizer.get_vocab_size()).to(device),
            labels.view(-1).to(device),
        ).to(device)
        valid_indicies = torch.where(
            labels.view(-1) == tokenizer.vocab.index(PADDING_TOKEN), False, True
        )
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        train_pred_probs = torch.softmax(train_predictions, dim=-1)
        train_pred_indices = torch.argmax(train_pred_probs, dim=-1)
        total_loss += loss.item()
        epoch_training_loss += loss.item()

    if (epoch + 1) % EVAL_INTERVAL == 0:
        model.eval()
        val_loss = 0
        # print(f"Epoch {epoch}")
        with torch.no_grad():
            iterator = iter(val_loader)
            for batch_num, batch in enumerate(iterator):
                input_batch, output_batch = batch
                input_batch, output_batch = input_batch.to(device), output_batch.to(
                    device
                )
                (
                    encoder_self_attention_mask,
                    decoder_self_attention_mask,
                    decoder_cross_attention_mask,
                ) = create_masks(input_batch, output_batch, tokenizer)
                optim.zero_grad()
                predictions = model(
                    input_batch,
                    output_batch,
                    encoder_self_attention_mask.to(device),
                    decoder_self_attention_mask.to(device),
                    decoder_cross_attention_mask.to(device),
                )
                labels = output_batch
                loss = criterian(
                    predictions.view(-1, tokenizer.get_vocab_size()).to(device),
                    labels.view(-1).to(device),
                ).to(device)
                valid_indicies = torch.where(
                    labels.view(-1) == tokenizer.vocab.index(PADDING_TOKEN), False, True
                )
                loss = loss.sum() / valid_indicies.sum()
                val_loss += loss.item()
                # Convert the predictions to probabilities
                pred_probs = torch.softmax(predictions, dim=-1)
                pred_indices = torch.argmax(pred_probs, dim=-1)
            # Print the predictions and corresponding ground truth
            print(f"Epoch {epoch} validation loss : {val_loss}")

    print("Input batch:", input_batch.tolist())
    print("Output batch:", output_batch.tolist())
    print("Predictions:", train_pred_indices.tolist())
    epoch_training_loss = epoch_training_loss / NUM_EPOCHS
    if (epoch + 1) % EVAL_INTERVAL == 0:
        print(f"Epoch {epoch} training loss : {epoch_training_loss}")
    graph_train_losses.append(epoch_training_loss)
    graph_val_losses.append(val_loss)

print(f"Total training loss : {total_loss}")

# save the model
save_directory = "saved_models"
os.makedirs(save_directory, exist_ok=True)
model_save_path = os.path.join(save_directory, "generate_chords_model.pth")
print(model_save_path)
print("here?")

torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# model.load_state_dict(torch.load("saved_models/generate_chords_model.pth"))

save_tokenizer(tokenizer, "tokenizer.json")

index = 0
graphs = [graph_train_losses]
# print(graph_val_losses)
# print(graph_train_losses)
graphs_string = ["Training Loss"]
for losses in graph_train_losses:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), losses, linestyle="-", color="black")
    plt.title(f"Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
    index += 1

    plt.savefig(
        os.path.join("output", f"loss_plot.png")
    )  # You can change the file format if needed
    plt.close()
