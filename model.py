##### model.py

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from typing import List, Tuple, Union, Any


class CharRNNModel(nn.Module):

    def __init__(self,
                 vocab_size=None,
                 embedding_dim=300,
                 hidden_dim=512,
                 num_layers=2,
                 dropout=0.2,
                 padding_value=0,
                 output_dim=None):
        super(CharRNNModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.padding_value = padding_value
        self.output_dim = output_dim

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_value)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True,
                            dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    def forward(self,
                inputs: torch.Tensor,  # inputs: numericalized tokens from smiles
                lengths: torch.Tensor,  # lengths from each numericalized tokens (sequence length)
                hiddens: Tuple[torch.Tensor] = None,
                **kwargs, ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:

        x = self.embeddings(inputs)
        x = rnn_utils.pack_padded_sequence(x,  # remove padding part
                                           lengths.cpu(),
                                           # lengths # GPU -> CPU for pack_padded_sequence calculation (better choice)
                                           batch_first=True,
                                           enforce_sorted=False)
        x, hiddens = self.lstm(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)  # restore padding part
        outputs = self.fc(x)

        return outputs, hiddens

    def reset_states(self, batch_size: int):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        return (h0, c0)

    @torch.no_grad()
    def generate(self,
                 tokenizer,
                 max_length: int = 64,
                 num_return_sequences: int = 1,
                 **kwargs):

        sos_token_id = tokenizer.char2int["<<SOS>>"]
        eos_token_id = tokenizer.char2int["<<EOS>>"]

        initial_inputs = torch.full((num_return_sequences, 1),
                                    sos_token_id,  # <<SOS>>: 1 # tokenizer.char2int(<<SOS>>)
                                    dtype=torch.long,
                                    device=self.device)

        generated_sequences = initial_inputs
        hiddens = self.reset_states(num_return_sequences)
        is_finished = torch.zeros(num_return_sequences, dtype=torch.bool, device=self.device)

        for _ in range(max_length):
            if is_finished.all():
                break

            # Forward pass
            x = self.embeddings(generated_sequences[:, -1:])  # Use the last generated token
            x, hiddens = self.lstm(x, hiddens)
            logits = self.fc(x)
            next_token_logits = logits.squeeze(1)

            # Sample the next token
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probabilities, num_samples=1)

            # Update finished sequences
            is_finished = is_finished | (next_tokens.squeeze(1) == eos_token_id)

            # Concatenate the new tokens
            generated_sequences = torch.cat([generated_sequences, next_tokens], dim=1)

        generated_smiles_list = []
        for sequence in generated_sequences:
            # Decode the sequence using the tokenizer
            # The tokenizer's decode method handles special tokens
            decoded_smiles = tokenizer.decode(sequence.tolist())
            generated_smiles_list.append(decoded_smiles)

        return generated_smiles_list


def count_parameters(model):
    """
    Calculates the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Calculate and print the total parameters
if __name__ == '__main__':
    my_deep_model = CharRNNModel()
    total_params = count_parameters(my_deep_model)
    print(f"Total trainable parameters: {total_params:,}")