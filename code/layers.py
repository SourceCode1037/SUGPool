from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch.nn import Parameter
import torch
import torch.nn as nn
import numpy
from torch_sparse import SparseTensor, sum, mul, fill_diag
import torch.nn.functional as F


class SUGPool(torch.nn.Module):
    def __init__(self, in_channels, args, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):
        super(SUGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.non_linearity = non_linearity

        self.lin_2 = nn.Linear(in_channels, 1)
        self.args = args

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        num_node = x.size(0)

        k = F.relu(self.lin_2(x))

        A = SparseTensor.from_edge_index(edge_index=edge_index, edge_attr=edge_attr, sparse_sizes=(num_node, num_node))
        I = SparseTensor.eye(num_node, device=self.args.device)
        A_wave = fill_diag(A, 1)

        s = A_wave @ k

        score = s.squeeze()
        perm = topk(score, self.ratio, batch)

        A = self.norm(A)

        K_neighbor = A * k.T
        x_neighbor = K_neighbor @ x

        # ----modified
        deg = sum(A, dim=1)
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        x_neighbor = x_neighbor * deg_inv.view(1, -1).T
        # ----
        x_self = x * k

        x = x_neighbor * (1 - self.args.combine_ratio) + x_self * self.args.combine_ratio

        x = x[perm]
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=s.size(0))

        return x, edge_index, edge_attr, batch, perm

    def norm(self, edge_index):
        adj = edge_index
        deg = sum(adj, dim=1)
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        adj = mul(adj, deg_inv.view((-1, 1)))
        return adj
