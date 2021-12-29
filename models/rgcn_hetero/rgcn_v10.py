"""
R-GCN from scratch
References:
- Tutorial: https://docs.dgl.ai/tutorials/models/1_gnn/4_rgcn.html
"""

import loguru
from functools import partial

import torch
import torch.nn as nn
import dgl.nn.functional as fn


# Convolutional Layer from scratch
class RelGraphConv(nn.Module):
    """
    Takes an input heterogeneous graph (w/ multiple edge types), and computes 1 round
    of convolution (message passing from neighbors + aggregation).
    If this is used as an input layer, then ??
    """

    def __init__(
        self, input_dim: int, output_dim: int, num_rels: int,
        num_bases: int = -1, activation=None,
        is_input_layer=False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.activation = activation
        self.is_input_layer = is_input_layer

        if not (0 < num_bases < num_rels):
            self.num_bases = num_rels

        # weight bases for the relations
        # NOTE: Init one learnable weight matrix per base, instead of one per relation
        #   for efficiency purposes; Construct the weight matrix for a relation as
        #   a linear combination of these bases later (w_comp)
        self.weight = nn.Parameter(torch.Tensor(
            self.num_bases, self.input_dim, self.output_dim))
        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        self._init_params()

    def _init_params(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.input_dim, self.num_bases, self.output_dim)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.input_dim, self.output_dim)
        else:
            weight = self.weight

        # QN: still a bit confused here, why there's a need to split. My guess is that input
        # layer has no h (node features)? If so, why not just create 'ones' as node features
        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be an embedding layer
                # using source node id
                # NOTE: edges.data['rel_type'] is a tensor of shape (input_dim,) with values between 0 and num_rels-1
                # NOTE: edges.data['norm'] is a tensor of shape (input_dim,1)
                embed = weight.view(-1, self.output_dim)
                index = edges.data['rel_type'] * self.input_dim + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:
            def message_func(edges):
                w = weight[edges.data['rel_type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                return {'msg': msg * edges.data['norm']}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class RGCN(nn.Module):
    def __init__(
        self, num_nodes,
    ) -> None:
        super().__init__()
        # INCOMPLETE
