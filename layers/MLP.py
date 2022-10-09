import torch
import torch.nn as nn


def FeedForward(dim, mult = 3, dropout = 0.):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
        nn.Dropout(dropout)
    )

class FeedForwardBlock(nn.Module):
    def __init__(self, dim, mult, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult, dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.post_norm(x + self.ff(x))
