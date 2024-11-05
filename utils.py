import math
import torch

# import torch.nn
from torch.nn import functional as F
import numpy as np

# GLOBAL VARIABLES
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 15
NUM_EPOCHS = 15
LR = 0.0001
EVAL_INTERVAL = 1  # how often you want to run val data

START_TOKEN = "<START>"
END_TOKEN = "<END>"
PADDING_TOKEN = "<PAD>"
MAX_SEQ_LEN = 180
NEG_INF = -1e10


d_model = 144  # dimension of model. you can play around with this for embedding,
# since vocab size for chords is much less than vocab size for the english language (dimensions would be 512). make this divisible by num_heads (8)
num_heads = 8  # number of attention heads
num_layers = 4  # number of encoder/decoder layers
dim_feedforward = d_model * 4  # dimension of feedforward network
dropout = 0.1  # dropout rate


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


# this implements the attention equation from Attention is All You Need
def self_attention(q, k, v, mask=None):
    # q, k, v are size batch_size * seq_len * num_heads * d_model/num_heads
    d_k = q.size()[-1]  # d_model/num_heads
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
        d_k
    )  # batch_size * num_heads * seq_len * seq_len
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(
        attention, v
    )  # batch_size * num_heads * seq_len * d_model/num_heads
    return values, attention


def create_masks(input_batch, output_batch, tokenizer):
    num_songs = len(input_batch)
    look_ahead_mask = torch.full([MAX_SEQ_LEN, MAX_SEQ_LEN], True)  # all trues
    look_ahead_mask = torch.triu(
        look_ahead_mask, diagonal=1
    )  # zeroes out elements below the main diagonal

    encoder_padding_mask = torch.full([num_songs, MAX_SEQ_LEN, MAX_SEQ_LEN], False)
    decoder_padding_mask_self_attention = torch.full(
        [num_songs, MAX_SEQ_LEN, MAX_SEQ_LEN], False
    )
    decoder_padding_mask_cross_attention = torch.full(
        [num_songs, MAX_SEQ_LEN, MAX_SEQ_LEN], False
    )

    for idx in range(num_songs):

        curr_input_song = input_batch[idx]
        curr_output_song = output_batch[idx]
        try:
            input_chord_size = len(
                curr_input_song[: tokenizer.vocab.index(PADDING_TOKEN)]
            )
        except ValueError:
            input_chord_size = len(curr_input_song)
        try:
            output_chord_size = len(
                curr_output_song[: tokenizer.vocab.index(PADDING_TOKEN)]
            )
        except ValueError:
            output_chord_size = len(curr_output_song)

        input_chords_to_padding_mask = np.arange(input_chord_size + 1, MAX_SEQ_LEN)
        output_chords_to_padding_mask = np.arange(output_chord_size + 1, MAX_SEQ_LEN)
        encoder_padding_mask[idx, :, input_chords_to_padding_mask] = True
        encoder_padding_mask[idx, input_chords_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, output_chords_to_padding_mask] = (
            True
        )
        decoder_padding_mask_self_attention[idx, output_chords_to_padding_mask, :] = (
            True
        )
        decoder_padding_mask_cross_attention[idx, :, input_chords_to_padding_mask] = (
            True
        )
        decoder_padding_mask_cross_attention[idx, output_chords_to_padding_mask, :] = (
            True
        )

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INF, 0)
    decoder_self_attention_mask = torch.where(
        look_ahead_mask + decoder_padding_mask_self_attention, NEG_INF, 0
    )
    decoder_cross_attention_mask = torch.where(
        decoder_padding_mask_cross_attention, NEG_INF, 0
    )
    return (
        encoder_self_attention_mask,
        decoder_self_attention_mask,
        decoder_cross_attention_mask,
    )


exclude_chords = [
    "Bmadd11",
    "F#m7add11",
    "Hm",
    "Ebm7",
    "Gsus2",
    "Fm7",
    "Bm7b5",
    "E7sus4",
    "Dmaj9",
    "A#m",
    "Ab7",
    "F#dim7",
    "Bbm7",
    "Dbmaj7",
    "Am6",
    "Ebm",
    "Gm6",
    "Emadd9",
    "Dm6",
    "B7sus4",
    "D9",
    "G#m7",
    "Gsus",
    "F#m7add11",
    "Em6",
    "Csus2",
    "F#5",
    "C6",
    "Gadd11",
    "Gadd9",
    "FM7",
    "A5",
    "Gb",
    "D7b9",
    "Csus4",
    "F6",
    "Cadd2",
    "Fm6",
    "C#dim",
    "Esus",
    "E5",
    "Gadd4",
    "A6",
    "Daug",
    "Fsus4",
    "Cm7",
    "Aadd9",
    "F5",
    "B5",
    "Fadd4",
    "Eb7",
    "C9",
    "E7b9",
    "Bb7b13",
    "Abmaj7",
    "Fadd#11",
    "F9",
    "C5",
    "Bmadd11",
    "EbM7",
    "Gbmaj7",
    "F#m7b5",
    "A9",
    "G#dim",
    "Dbadd9",
    "D2",
    "Asus",
    "A#",
    "C9sus4",
    "G#mmaj7",
    "Bb5",
    "D#",
    "C#5",
    "G9",
    "F7sus",
    "G6add4",
    "Fdim",
    "Ebdim",
    "Ammaj7",
    "Cmaj9",
    "Gaug",
    "Bbm6",
    "Fdim7",
    "Fmaj7sus2",
    "Fmaj9",
    "F#madd9",
    "Bbsus2",
    "Ddim",
    "C#sus4",
    "E9",
    "Ebsus2",
    "Gdim",
    "F#7add11",
    "Csus",
    "F#dim",
    "Fsus",
    "Amaj9b5",
    "Em#5",
    "DM7",
    "A#aug",
    "Abm",
    "Eadd9",
    "B2",
    "D7sus4",
    "Bbaug",
    "Ebaug",
    "E7sus2",
    "CM7",
    "Bdim7",
    "G2",
    "Dmmaj7",
    "Eaug",
    "D3",
    "Dadd4",
    "Gdim7",
    "A7sus",
    "Caug",
    "Dm9",
    "Badd9",
    "Gbdim7",
    "F13",
    "D9sus4",
    "Gb7",
    "Fmaj7#11",
    "Cdim",
    "D#b5",
    "Ab6",
    "Db6",
    "B9sus",
    "Bbdim7",
    "Bb9",
    "G7sus4",
    "C3",
    "Gm9",
    "Bb13",
    "Abm7b5",
    "Amadd9",
    "C7sus4",
    "Dm7b5",
    "Dbdim",
    "Bdim",
    "E6",
    "Db9",
    "Esus2",
    "Edim7",
    "F#m9",
    "Daug5",
    "Fb5",
    "Eb6",
    "Em7b5",
    "E7#9",
    "Dm7add11",
    "Badd11",
    "Gmadd11",
    "Bbmmaj9",
    "Ebmadd9",
    "Ebadd9",
    "Cm7add4",
    "A7b13",
    "Bmaj7",
    "Dbdim7",
    "EM7",
    "Cm6",
    "D#maj7",
    "F6sus2",
    "F7sus4",
    "Cmaj7#5",
    "D#dim",
    "Eb5",
    "Bbm9",
    "Eb7b9",
    "Gmmaj7",
    "Db7",
    "D7#9",
    "A#dim",
    "G#7sus4",
    "Bmadd4",
    "F#sus",
    "EmM7",
    "Bbadd9",
    "D6sus2",
    "G#dim7",
    "G6sus2",
    "Bbm7b5",
    "F7b9",
    "Fma7",
    "D7#5",
    "D7sus2",
    "Edim",
    "Fm7b5",
    "Gbm7",
    "Abm7",
    "C13",
    "C7b9",
    "F#7#9",
    "Am7add11",
    "C#m7b5",
    "Fadd9#11",
    "Aaug",
    "B7b9",
    "B7#5",
    "C#sus2",
    "Hm",
    "E7sus",
    "Dmadd4",
    "D#5",
    "Daddb6",
    "Gmadd9",
    "F#sus4",
    "A7#5",
    "Faug",
    "Am7b5",
    "Bm9",
    "Ddim7",
    "Dbm7",
    "C#maj7",
    "AM7",
    "G#sus4",
    "Adim",
    "Am9",
    "F9sus4",
    "Ebmaj",
    "Db7b9",
    "Bb7b9",
    "C4",
    "Abm6",
    "Gm7b5",
    "G6sus4",
    "B#m",
    "Gmaj",
    "Dadd4add9",
    "F#7sus4",
    "Emaug5",
    "Gb9",
    "Ab9",
    "Bm6",
    "Ebm6",
    "Cb",
    "Gm7add11",
    "C#9#5",
]
