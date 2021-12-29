from collections import defaultdict

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class RelGraphConv(nn.Module):
    def __init__(
        self, input_dim, output_dim,
        rel_names, num_bases, *,
        weight=True, bias=True, activation=None,
        self_loop=False, dropout=0.,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.GraphConv(input_dim, output_dim, norm='right', weight=False, bias=False)
            for rel in rel_names
        }, aggregate='sum')
        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dgl.nn.WeightBasis((input_dim, output_dim), num_bases, len(self.rel_names))
