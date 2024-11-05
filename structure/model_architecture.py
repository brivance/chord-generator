import torch
from utils import device, self_attention
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        # get a range of 0 to the model dimensions, skipping 2 (even numbers only)
        evens = torch.arange(0, self.d_model, 2).float()

        # this is the equation from Attention is All You Need for positional encoding, using sin and cos functions
        # you only need to find evens (not odds) because the denominator for even and odds turn out to be the same
        denominator = torch.pow(10000, evens / self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(
            self.max_sequence_length, 1
        )
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)

        # interweave the two matrices (first indice from the even, second from odd, third from even, so on)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE


class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"

    def __init__(self, max_sequence_length, d_model, vocab):
        super().__init__()
        self.vocab_size = len(vocab)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        # self.vocab = vocab
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):  # sentence
        # pdb.set_trace()
        x = self.embedding(x)
        pos = self.position_encoder().to(device)
        x = self.dropout(x + pos)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, sequence_length, d_model = (
            x.size()
        )  # batch_size * seq_len * d_model
        qkv = self.qkv_layer(x)  # batch_size * seq_len * (3*d_model)
        qkv = qkv.reshape(
            batch_size, sequence_length, self.num_heads, 3 * self.head_dim
        )  # batch_size * seq_len * num_heads * (3*d_model/num_heads)
        qkv = qkv.permute(
            0, 2, 1, 3
        )  # batch_size * num_heads * seq_len * (3*d_model/num_heads)
        q, k, v = qkv.chunk(
            3, dim=-1
        )  # each q,k,v: batch_size * num_heads * seq_len * d_model/num_heads
        values, attention = self_attention(
            q, k, v, mask
        )  # values are batch_size * num_heads * seq_len * n_model/num_heads
        values = values.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )  # values.size() should be the same as x.size()
        out = self.linear_layer(values)  # batch_size * seq_len * d_model
        return out


class LayerNormalization(nn.Module):
    # layer normalization is normalization performed on every part of one feature (layer)
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape  # typically this should be your model dimension (d_model), ex. [512]
        self.eps = eps  # making sure your denominator is never 0
        self.gamma = nn.Parameter(
            torch.ones(parameters_shape)
        )  # learnable parameter. [d_model]
        self.beta = nn.Parameter(
            torch.zeros(parameters_shape)
        )  # learnable parameter. [d_model]

    def forward(
        self, inputs
    ):  # input shape: batch_size * seq_len * d_model . input/output shape is the same
        # layer normalization is mostly for back prop and being able to learn the parameters of beta and gamma
        dims = [
            -(i + 1) for i in range(len(self.parameters_shape))
        ]  # perform layer normalization on last shape dimension, which is d_model
        mean = inputs.mean(
            dim=dims, keepdim=True
        )  # batch_size * seq_len * 1 (the mean is 1 dim)
        var = ((inputs - mean) ** 2).mean(
            dim=dims, keepdim=True
        )  # calculate variance and standard deviation
        std = (var + self.eps).sqrt()
        y = (
            inputs - mean
        ) / std  # batch_size * seq_len * d_model, brings last dimension back to d_model
        out = self.gamma * y + self.beta
        return out  # batch_size * seq_len * d_model


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(
        self, x, y, mask
    ):  # you can find shapes of this function at 2:48 of the 3 hr video
        batch_size, sequence_length, d_model = (
            x.size()
        )  # batch_size * seq_len * d_model
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = self_attention(
            q, k, v, mask
        )  # We don't need the mask for cross attention, removing in outer function!
        values = values.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, d_model
        )
        out = self.linear_layer(values)
        return out  # batch_size * seq_len * d_model


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=dropout)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, ffn_hidden=ffn_hidden, dropout=dropout
        )
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, self_attention_mask):
        residual_x = x.clone()  # batch_size * max_seq_len * d_model
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x  # shape should stay consistent through whole forward pass (it will change within some functions, but not on this level)


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model,
        ffn_hidden,
        num_heads,
        dropout,
        num_layers,
        max_sequence_length,
        vocab,
    ):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, vocab)
        self.layers = SequentialEncoder(
            *[
                EncoderLayer(d_model, ffn_hidden, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, self_attention_mask):
        x = self.sentence_embedding(x)
        x = self.layers(x, self_attention_mask)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, ffn_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden)  # d_model * dim_feedforward
        self.linear2 = nn.Linear(ffn_hidden, d_model)  # d_model * d_model
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):  # batch * seq_len * d_model
        x = self.linear1(x)  # batch * seq_len * dim_feedforward
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)  # batch * seq_len * d_model
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=dropout)

        self.encoder_decoder_attention = MultiHeadCrossAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=dropout)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, ffn_hidden=ffn_hidden, dropout=dropout
        )
        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = (
            y.clone()
        )  # batch_size * seq_len * d_model, should be consistent throughout forward function
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)
        return y  # batch_size * seq_len * d_model


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y


class Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        ffn_hidden,
        num_heads,
        dropout,
        num_layers,
        max_sequence_length,
        vocab,
    ):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, vocab)
        self.layers = SequentialDecoder(
            *[
                DecoderLayer(d_model, ffn_hidden, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        # x and y are batch_size * seq_len * d_model
        # mask is seq_len * seq_len
        y = self.sentence_embedding(y)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        ffn_hidden,
        num_heads,
        dropout,
        num_layers,
        max_sequence_length,
        vocab_size,
        vocab,
    ):
        super().__init__()
        self.encoder = Encoder(
            d_model,
            ffn_hidden,
            num_heads,
            dropout,
            num_layers,
            max_sequence_length,
            vocab,
        )
        self.decoder = Decoder(
            d_model,
            ffn_hidden,
            num_heads,
            dropout,
            num_layers,
            max_sequence_length,
            vocab,
        )
        self.linear = nn.Linear(d_model, vocab_size)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def forward(
        self,
        x,
        y,
        encoder_self_attention_mask=None,
        decoder_self_attention_mask=None,
        decoder_cross_attention_mask=None,
    ):  # x, y are batch of sentences
        x = self.encoder(x, encoder_self_attention_mask)
        out = self.decoder(
            x, y, decoder_self_attention_mask, decoder_cross_attention_mask
        )
        out = self.linear(out)
        return out
