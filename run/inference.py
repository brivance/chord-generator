import torch

import sys
import os

# Get the parent directory of the current file's directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from structure.model_architecture import Transformer
from utils import (
    END_TOKEN,
    device,
    d_model,
    dim_feedforward,
    num_heads,
    dropout,
    num_layers,
    MAX_SEQ_LEN,
)
from structure.tokenizer import load_tokenizer

PADDING_TOKEN_ID = 2

tokenizer = load_tokenizer("tokenizer.json")

saved_model = Transformer(
    d_model,
    dim_feedforward,
    num_heads,
    dropout,
    num_layers,
    MAX_SEQ_LEN,
    tokenizer.get_vocab_size(),
    tokenizer.vocab,
)

saved_model.load_state_dict(torch.load("saved_models/generate_chords_model.pth"))

# ensure the model is in evaluation mode
saved_model.eval()
saved_model = saved_model.to(device)


# def run_inference(model, input_tensor, output_tensor):
#     # move tensors to the correct device
#     input_tensor = input_tensor.to(device)
#     output_tensor = output_tensor.to(device)

#     with torch.no_grad():  # no need to compute gradients for inference
#         predictions = model(input_tensor, output_tensor)

#     return predictions


def post_process_output(predictions, tokenizer):
    # convert predictions to indices
    predicted_indices = predictions.argmax(dim=-1).squeeze().tolist()

    # convert indices to tokens
    predicted_tokens = tokenizer.decode_sequence(predicted_indices)
    return predicted_tokens


input_chords = ["E", "A"]
input_tokens = tokenizer.tokenize(
    input_chords, start_token=True, end_token=False
)  # add start token only for initial input
print(input_tokens)
input_tensor = (
    torch.tensor(input_tokens).unsqueeze(0).to(device)
)  # add batch dimension and move to device


def add_token(generated_tokens, token):
    try:
        padding_index = generated_tokens.index(PADDING_TOKEN_ID)
        generated_tokens[padding_index] = token

        # generated_tokens = generated_tokens[: padding_index + 1] + [
        #     PADDING_TOKEN_ID
        # ] * (len(generated_tokens) - padding_index - 1)

    except ValueError:
        pass

    return generated_tokens


def run_autoregressive_inference(model, input_tensor, seq_len, max_length, tokenizer):
    # pdb.set_trace()
    output_tensor = torch.zeros_like(input_tensor)
    output_tokens = output_tensor.squeeze(0).tolist()

    generated_tokens = input_tensor.squeeze(
        0
    ).tolist()  # initialize with the input tokens
    print(generated_tokens)

    with torch.no_grad():  # no gradients needed for inference
        for _ in range(max_length - seq_len):
            # convert current sequence into a tensor with batch dimension
            input_tensor = torch.tensor(generated_tokens).unsqueeze(0).to(device)
            output_tensor = torch.tensor(output_tokens).unsqueeze(0).to(device)

            # model predicts logits for the next token in sequence
            predictions = model(input_tensor, output_tensor)

            # get the token with the highest probability for the next step
            next_token = predictions[:, -1, :].argmax(dim=-1).item()
            print("next token", next_token)

            # append predicted token to the generated sequence
            add_token(generated_tokens, next_token)
            print(generated_tokens)

            # check for an end token (optional)
            if next_token == END_TOKEN:
                break

            seq_len += 1

    # convert token indices back to chord names
    predicted_chords = tokenizer.decode_sequence(generated_tokens)
    return predicted_chords


max_sequence_length = 20  # You can set this to whatever length you want

predicted_chords = run_autoregressive_inference(
    saved_model, input_tensor, len(input_chords) + 1, max_sequence_length, tokenizer
)
print(predicted_chords[: len(input_chords) + max_sequence_length])


# predictions = run_inference(saved_model, input_tensor, output_tensor)
# print(predictions)
# predicted_tokens = post_process_output(predictions, tokenizer)
# print(predicted_tokens)
