import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositionalTransformer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, num_heads,
                 basic_functions):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.basic_functions = basic_functions
        self.func_embeddings = nn.Embedding(len(basic_functions), hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                   nhead=num_heads,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, composition):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.size(0)

        h = self.input_proj(x)
        h = h.unsqueeze(1)

        comp_indices = torch.tensor(
            [self.basic_functions.index(f) for f in composition]).to(x.device)
        comp_embed = self.func_embeddings(comp_indices)
        comp_embed = comp_embed.unsqueeze(0).repeat(batch_size, 1, 1)

        h_seq = torch.cat([h, comp_embed], dim=1)
        h_encoded = self.transformer_encoder(h_seq)

        output = self.output_proj(h_encoded[:, -1, :])

        return output


class ISCompositionalTransformer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, num_heads,
                 basic_functions, max_comp_len):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.basic_functions = basic_functions

        self.func_embeddings = nn.Embedding(len(basic_functions), hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_enc = nn.Parameter(
            torch.randn(1, max_comp_len + 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                   nhead=num_heads,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)

        self.intermediate_proj = nn.Linear(hidden_dim, input_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, compositions):
        batch_size = x.size(0)
        device = x.device

        h = self.input_proj(x).unsqueeze(1)

        max_comp_len = max(len(comp) for comp in compositions)
        comp_embed_tensor = torch.zeros(batch_size,
                                        max_comp_len,
                                        self.hidden_dim,
                                        device=device)

        for i, comp in enumerate(compositions):
            indices = torch.tensor(
                [self.basic_functions.index(f) for f in comp], device=device)
            comp_embed_tensor[i, :len(comp), :] = self.func_embeddings(indices)

        h_seq = torch.cat([h, comp_embed_tensor], dim=1)
        h_seq += self.positional_enc[:, :h_seq.size(1), :]
        h_encoded = self.transformer_encoder(h_seq)

        intermediate_preds = self.intermediate_proj(h_encoded[:, 1:-1, :])

        final_pred = self.output_proj(h_encoded[:, -1, :])

        return intermediate_preds, final_pred
