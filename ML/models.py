import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Encoder-Decoder Neural Network for unsupervised anomaly detection.
    Learns to compress and reconstruct normal background network traffic patterns.
    """
    def __init__(self, input_dim):
        super().__init__()
        # Encoder: Compresses 12-dimensional features down to an 8-dimensional bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)  # Bottleneck layer
        )
        # Decoder: Attempts to reconstruct the compressed representation back to 12 dimensions
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))