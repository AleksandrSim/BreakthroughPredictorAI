import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        output_dim,
        num_time_varying_vars,
        dropout=0.1,
    ):
        super(SimpleTransformer, self).__init__()
        self.embedding_time_varying = nn.Linear(num_time_varying_vars, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x_time_varying):
        x_time_varying_embed = self.embedding_time_varying(x_time_varying)
        transformer_output = self.transformer_encoder(x_time_varying_embed)
        # Use the last hidden state for classification
        transformer_output = transformer_output[:, -1, :]
        output = self.fc_out(transformer_output)
        return output


# Test the simplified model
if __name__ == "__main__":
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    output_dim = 1
    num_time_varying_vars = 711
    dropout = 0.1

    model = SimpleTransformer(
        embed_dim, num_heads, num_layers, output_dim, num_time_varying_vars, dropout
    )

    # Dummy input
    x_time_varying = torch.randn(
        64, 32, num_time_varying_vars
    )  # (batch_size, seq_len, num_time_varying_vars)
    output = model(x_time_varying)
    print(output.shape)  # Expected output shape: (32, 1)
