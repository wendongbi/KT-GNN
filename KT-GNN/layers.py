from builtins import NotImplementedError
from functools import reduce
import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch import Tensor
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul, set_diag
from torch_geometric.nn.conv import MessagePassing, gat_conv, gcn_conv, sage_conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import SplineConv, GATConv, GATv2Conv, SAGEConv, GCNConv, GCN2Conv, GENConv, DeepGCNLayer, APPNP, JumpingKnowledge, GINConv
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from typing import Union, Tuple, Optional
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch_sparse

# complete_layer
from email import message
from sqlite3 import adapt
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
import copy


class Adapted_complete_layer(MessagePassing):
    def __init__(self, dim_share: int, dim_unshare: int, hidden: int = 64, 
                 normalize: bool = False, adapted: bool = False, 
                 bias: bool = True, dropout: float = 0.5, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'add')
        super(Adapted_complete_layer, self).__init__(**kwargs)

        self.normalize = normalize
        self.hidden = hidden

        self.adapted = adapted
        self.dim_share = dim_share
        self.dim_unshare = dim_unshare
        self.dropout = dropout

        # self.lin_f_src = Linear(dim_share, hidden, bias=False)
        # self.lin_f_tar = Linear(dim_share, hidden, bias=False)
        # # self.lin_f_a = Linear(hidden * 2, in_channels-dim_share, bias=False) # per-dimensional attention
        # self.lin_f_a = Linear(hidden * 2, 1, bias=False) # per-dimensional attention

        # 1-dim attention: GAT simplified version
        self.lin_f_src = nn.Sequential(
            Linear(dim_share, 1, bias=False),
            # nn.ReLU(),
            # Linear(hidden, 1, bias=False)
        )
        self.lin_f_tar = nn.Sequential(
            Linear(dim_share, 1, bias=False),
            # nn.ReLU(),
            # Linear(hidden, 1, bias=False)
        )

        if self.adapted:
            # self.lin_g = Linear(self.dim_share * 2, in_channels-dim_share, bias=False)
            self.lin_g = Linear(dim_unshare * 2, dim_unshare, bias=False)
            self.lin_diff = Linear(dim_share, dim_unshare)

        self.reset_parameters()

    def reset_parameters(self):
        # self.lin_f_src.reset_parameters()
        # self.lin_f_tar.reset_parameters()
        # self.lin_f_a.reset_parameters()
        for l in self.lin_f_src:
            if isinstance(l, Linear):
                l.reset_parameters()
        for l in self.lin_f_tar:
            if isinstance(l, Linear):
                l.reset_parameters()
        
        if self.adapted:
            self.lin_g.reset_parameters()
            self.lin_diff.reset_parameters()
    
    def g(self, x_u, domain_diff, activation=None):
        # Domain difference function
        # x_u: edge_index.shape[0] * dim_unshare
        # domain_diff: dim_share
        N = x_u.shape[0]
        # support = torch.cat((x_src[:, :self.dim_share], deltaX.unsqueeze(0).expand((N, self.dim_share))), dim=1)
        adapted_domain_diff = self.lin_diff(domain_diff.unsqueeze(0)) # dim_share -> 1 * dim_unshare
        support = self.lin_g(torch.cat((x_u, adapted_domain_diff.expand((N, self.dim_unshare))), dim=1))
        # support = support * adapted_domain_diff
        if activation is None:
            domain_shift = support
        elif activation == 'relu':
            domain_shift = F.relu(support)
        elif activation == 'tanh':
            domain_shift = F.tanh(support)
        else:
            raise NotImplementedError('Not implemented activation func:{}'.format(activation))
        return domain_shift, adapted_domain_diff # relu, tanh, none
    
    def f(self, x_share, edge_index):
        # Neighbor importance function
        # suppose that x_tar has been calibrated by the domain difference function g()
        alpha_src = self.lin_f_src(x_share)
        alpha_tar = self.lin_f_tar(x_share)
        e = alpha_src[edge_index[0]] + alpha_tar[edge_index[1]]
        return F.leaky_relu(e, negative_slope=0.1)

    def forward(self, x_o: Tensor, x_u: Tensor, edge_index: Adj, deltaX: Tensor=None, mask_source_node: Tensor=None, 
                size: Size = None) -> Tensor:
        """"""
        a_f = self.f(x_o, edge_index)
        adapted_domain_diff = None
        if self.adapted:
            domain_shift, adapted_domain_diff = self.g(x_u, deltaX, activation=None)
            message_u = x_u - domain_shift * mask_source_node.unsqueeze(-1)
        else:
            message_u = x_u
        x_u_hat = self.propagate(edge_index, x=message_u, a_f=a_f, size=size)
        # out = torch.cat((x[:, :self.dim_share], x_hat[:, self.dim_share:]), dim=1)
        # return torch.cat((x_o, x_u_hat), dim=1)
        return x_u_hat, adapted_domain_diff

    def message(self, x_j: Tensor, x_i: Tensor, a_f: Tensor,
            index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        # x_j is the source nodes
        # softmax要做么，怎么保证数量级统一
        alpha = a_f
        # print(index, ptr, index.shape, size_i)
        alpha = softmax(alpha, index, ptr, size_i) # edge_softmax
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha
        # return x_j
    
    # def message_and_aggregate(self, adj_t: SparseTensor,
    #                           x: OptPairTensor) -> Tensor:
    #     adj_t = adj_t.set_value(None, layout=None)
    #     return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(dim_share:{}, dim_unshare:{})'.format(self.__class__.__name__, self.dim_share,
                                   self.dim_unshare)



from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
class AdaptedConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, activation_g = None, negative_slope=0.1,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'add')
        super(AdaptedConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.activation_g = activation_g
        self.negative_slope = negative_slope

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        
        self.lin_s = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_t = Linear(in_channels[0], out_channels, bias=bias)
        
        self.a_g_s2t = Linear(in_channels[0] * 2, 1, bias=False)
        self.a_g_t2s = Linear(in_channels[0] * 2, 1, bias=False)
        self.a_f_s2t = Linear(out_channels, 1, bias=False)
        self.a_f_t2s = Linear(out_channels, 1, bias=False)
        

        self.reset_parameters()

    def reset_parameters(self):
        # self.lin_l.reset_parameters()
        self.lin_s.reset_parameters()
        self.lin_t.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
        self.a_g_s2t.reset_parameters()
        self.a_g_t2s.reset_parameters()
        self.a_f_s2t.reset_parameters()
        self.a_f_t2s.reset_parameters()
        

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_index1: Adj, edge_index2: Adj, central_mask: Tensor,
                size: Size = None) -> Tensor:
        """
            x: node faeture
            edge_index: all edges
            edge_index1: edges target at source-domain nodes
            edge_index2: edges target at target-domain nodes
        """
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        x_src = x[0] # (N, D)
        # g
        domain_diff = x_src[central_mask].mean(0, keepdim=True) - x_src[~central_mask].mean(0, keepdim=True) # (1, D)
        domain_diff = domain_diff.expand(x_src.shape) # (N, D)
        domain_shift_s2t = torch.tanh(self.a_g_s2t(torch.cat((x_src, domain_diff), dim=-1))) * domain_diff
        domain_shift_t2s = torch.tanh(self.a_g_t2s(torch.cat((x_src, domain_diff), dim=-1))) * domain_diff
        x_s2t = x_src - domain_shift_s2t * central_mask.unsqueeze(-1)
        x_t2s = x_src + domain_shift_t2s * (~central_mask).unsqueeze(-1)

        # f
        x_s2t = self.lin_t(x_s2t)
        x_t2s = self.lin_s(x_t2s)
        # # gat-v1 attention
        # a_s2t = self.a_f_s2t(x_s2t) # (N, 1)
        # a_t2s = self.a_f_t2s(x_t2s) # (N, 1)
        # alpha1 = F.leaky_relu(a_t2s[edge_index1[0]] + a_t2s[edge_index1[1]], negative_slope=self.negative_slope) # attention for edge_index1, target at source domain
        # alpha2 = F.leaky_relu(a_s2t[edge_index2[0]] + a_s2t[edge_index2[1]], negative_slope=self.negative_slope) # attention for edge_index2, target at target domain
        
        # gat-v2 attention
        a_t2s = F.leaky_relu(x_t2s[edge_index1[0]] + x_t2s[edge_index1[1]], negative_slope=self.negative_slope) 
        a_s2t = F.leaky_relu(x_s2t[edge_index2[0]] + x_s2t[edge_index2[1]], negative_slope=self.negative_slope) 
        alpha1 = self.a_f_t2s(a_t2s) # attention for edge_index1, target at source domain
        alpha2 = self.a_f_s2t(a_s2t) # attention for edge_index2, target at target domain
        
        # sparse-implemented softmax for [a1 || a2]
        alpha = torch.cat((alpha1, alpha2), dim=0)
        alpha = softmax(alpha, edge_index[1], ptr=None, num_nodes=x_src.shape[0])
        

        # propagate_type: (x: OptPairTensor)
        out_t2s = self.propagate(edge_index1, x=x_t2s, size=size, alpha=alpha[:alpha1.shape[0]]) # message passing from target to source domain
        out_s2t = self.propagate(edge_index2, x=x_s2t, size=size, alpha=alpha[alpha1.shape[0]:]) # message passing from source to tar domain
        out = out_t2s + out_s2t
        # out = self.lin_l(out)
        # print(x_src.shape, x[1].shape, out.shape)
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, alpha) -> Tensor:
        # print(x_j.shape, alpha.shape)
        return x_j * alpha

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)