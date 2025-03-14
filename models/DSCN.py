import torch
import torch.nn as nn
import pytorch_lightning as pl

class DeepSubspaceClusteringNetwork(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(DeepSubspaceClusteringNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)