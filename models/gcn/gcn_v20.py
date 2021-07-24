"""
GCN using DGL's pre-built GNN modules.

Symbols used:
- N: number of nodes
- F: node feature vector dimensions
- H: hidden node feature vector dimensions
- C: number of classes in the classification problem
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """Simple graph convolutional network with 2 graph convolutional layers,
    for node classification.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(input_dim, hidden_dim)
        self.conv2 = dgl.nn.GraphConv(hidden_dim, num_classes)

    def forward(self, g, in_feats: torch.Tensor) -> torch.Tensor:
        h = self.conv1(g, in_feats)  # (N, F) -> (N, H)
        h = F.relu(h)
        h = self.conv2(g, h)  # (N, H) -> (N, C)
        return h
