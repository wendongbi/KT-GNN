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
from layers import Adapted_complete_layer, AdaptedConv


class Adapted_complementor(nn.Module):
    def __init__(self, dim_o, dim_u, hidden_o=128, hidden_u=128, step=2, use_dist_loss=False, use_complement=True):
        super(Adapted_complementor, self).__init__()
        self.dim_o = dim_o
        self.dim_u = dim_u
        self.hidden_o = hidden_o
        self.hidden_u = hidden_u
        self.step = step
        self.input_layer_o = Linear(dim_o, hidden_o, bias=False)
        self.input_layer_u = Linear(dim_u, hidden_u, bias=False)
        self.use_complement = use_complement
        if use_complement:
            self.adapted_layer = Adapted_complete_layer(hidden_o, hidden_u, adapted=True, bias=False)
            self.use_dist_loss = use_dist_loss
            self.layers = nn.ModuleList()
            for _ in range(step-1):
                self.layers.append(Adapted_complete_layer(hidden_o, hidden_u, adapted=False, bias=False))
        self.graph_partitioned = None
        self.reset_parameters()

    def reset_parameters(self):
        self.input_layer_u.reset_parameters()
        self.input_layer_o.reset_parameters()
        if self.use_complement:
            self.adapted_layer.reset_parameters()
            for layer in self.layers:
                layer.reset_parameters()
    
    def prepare_graph(self, data, step=2):
        edge_index = data.edge_index
        mask_src = copy.deepcopy(data.central_mask)
        mask_tar = ~mask_src
        graphs = []
        for _ in range(step):
            mask_e = mask_src[edge_index[0]] * mask_tar[edge_index[1]]
            new_src_ids = torch.Tensor(list(set(edge_index[1, mask_e].tolist()))).long()
            # print(mask_src.sum(), mask_tar.sum(), len(new_src_ids))
            graphs.append((edge_index[:, mask_e], torch.where(mask_src)[0], new_src_ids))
            # update mask_src and mask_tar
            mask_tar[new_src_ids] = False
            mask_src[:] = False
            mask_src[new_src_ids] = True
        cnt = 0
        for idx, graph in enumerate(graphs):
            g, _, _ = graph
            cnt += g.shape[1]
            print(idx, g.shape)
        return graphs
    
    def distribution_loss(self, adapted_domain_diff, x_u_hat, central_mask, tar_mask):
        new_domain_diff = x_u_hat[central_mask].mean(0, keepdim=True) - x_u_hat[tar_mask].mean(0, keepdim=True)
        return F.mse_loss(new_domain_diff, adapted_domain_diff)


    def forward(self, data):
        # print(self.training)
        if self.graph_partitioned is None:
            self.graph_partitioned = self.prepare_graph(data, step=self.step)
        x_o = self.input_layer_o(data.x[:, :self.dim_o])
        x_u = self.input_layer_u(data.x[:, self.dim_o:])
        if not self.use_complement:
            return torch.cat((x_o, x_u), dim=1), None
        deltaX = x_o[data.central_mask].mean(0) - x_o[~data.central_mask].mean(0)
        x_u_hat, adapted_domain_diff = self.adapted_layer(x_o, x_u, self.graph_partitioned[0][0], deltaX, data.central_mask)
        if self.training and self.use_dist_loss:
            loss_dist = self.distribution_loss(adapted_domain_diff, x_u_hat, data.central_mask, self.graph_partitioned[0][2])
        else:
            loss_dist = None
        for idx, layer in enumerate(self.layers):
            x_u_hat, _ = layer(x_o, x_u_hat, self.graph_partitioned[idx+1][0]) # change data.edge_index further
        return torch.cat((x_o, x_u*data.central_mask.unsqueeze(1) + x_u_hat* (~data.central_mask).unsqueeze(1)), dim=1), loss_dist


class KTGNN(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, hidden=64, root_weight=False, 
     dim_share=300, step=1, hidden_o=128, hidden_u=128, use_dist_loss=False, cached_edges=True,
     dropout=0.5, use_bn=False):
        super(KTGNN, self).__init__()
        self.cached_edges = cached_edges
        self.dropout = dropout
        self.use_bn = use_bn
        self.convs = nn.ModuleList()
        self.complementor = Adapted_complementor(dim_o=dim_share, dim_u=dataset.num_features - dim_share, hidden_o=hidden_o, hidden_u=hidden_u, step=step, use_dist_loss=use_dist_loss)
        self.bns = torch.nn.ModuleList()
        if layer_num == 1:
            self.convs.append(
                AdaptedConv(hidden_o+hidden_u, dataset.num_classes, root_weight=root_weight)
            )
        else:
            for num in range(layer_num-1):
                if num == 0:
                    self.convs.append(AdaptedConv(hidden_o+hidden_u, hidden, root_weight=root_weight))
                    if self.use_bn:
                        self.bns.append(torch.nn.BatchNorm1d(hidden))
                # elif num == layer_num - 1:
                #     self.convs.append(AdaptedConv(hidden, dataset.num_classes, root_weight=root_weight))
                else:
                    self.convs.append(AdaptedConv(hidden, hidden, root_weight=root_weight))
                    if self.use_bn:
                        self.bns.append(torch.nn.BatchNorm1d(hidden))
        self.clf_base = AdaptedConv(hidden, dataset.num_classes, root_weight=root_weight)
        hidden_target = hidden
        self.clf_target = AdaptedConv(hidden_target, dataset.num_classes, root_weight=root_weight)
        self.clf_transformer = nn.Sequential(
            Linear(hidden, hidden, bias=True),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            Linear(hidden, hidden_target, bias=True)
        )
        if cached_edges:
            self.edge_index1, self.edge_index2, self.edge_index = None, None, None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.complementor.reset_parameters()
        self.clf_base.reset_parameters()
        self.clf_target.reset_parameters()
        for l in self.clf_transformer:
            if isinstance(l, Linear) or isinstance(l, nn.BatchNorm1d):
                l.reset_parameters()
    
    def graph_partition(self, edge_index, central_mask, add_self_loop=True):
        if add_self_loop:
            # add self loop
            assert isinstance(edge_index, Tensor)
            num_nodes = central_mask.shape[0]
            edge_index, edge_attr = remove_self_loops(
                edge_index)
            edge_index, edge_attr = add_self_loops(
                edge_index, fill_value='mean',
                num_nodes=num_nodes)
        mask1 = central_mask[edge_index[1]]
        mask2 = (~central_mask)[edge_index[1]]
        edge_index1, edge_index2 = edge_index[:, mask1], edge_index[:, mask2]
        return edge_index1, edge_index2, torch.cat((edge_index1, edge_index2), dim=-1)
    

    def forward(self, data):
        # x, edge_index = data.x, data.edge_index
        x, loss_dist = self.complementor(data)
        central_mask = data.central_mask
        if self.cached_edges:
            if self.edge_index is None:
                self.edge_index1, self.edge_index2, self.edge_index = self.graph_partition(data.edge_index, central_mask)
            edge_index1, edge_index2, edge_index = self.edge_index1, self.edge_index2, self.edge_index
        else:
            edge_index1, edge_index2, edge_index = self.graph_partition(data.edge_index, central_mask)
        # print(edge_index1.shape, edge_index2.shape, edge_index.shape)
        
        # adj_sp = torch_sparse.SparseTensor(row=edge_index[1], col=edge_index[0], value=torch.ones(edge_index.shape[1]).to(x.device), sparse_sizes=(x.shape[0], x.shape[0]))
        for ind, conv in enumerate(self.convs):
            # if ind == len(self.convs) -1:
            #     x = conv(x, edge_index, edge_index1, edge_index2, central_mask)
            # else:

            # x = F.elu(conv(x, edge_index, edge_index1, edge_index2, central_mask))
            x = conv(x, edge_index, edge_index1, edge_index2, central_mask)
            if self.use_bn:
                x = self.bns[ind](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
                # x = conv(x, edge_index)
        logits_base = self.clf_base(x, edge_index, edge_index1, edge_index2, central_mask)
        logits_transformed_target = self.clf_target(self.clf_transformer(x), edge_index, edge_index1, edge_index2, central_mask)
        logits_target = self.clf_target(x, edge_index, edge_index1, edge_index2, central_mask)
        return F.log_softmax(logits_base, dim=1),  F.log_softmax(logits_target, dim=1), F.log_softmax(logits_transformed_target, dim=1), loss_dist
    def get_emb(self, data):
        # x, edge_index = data.x, data.edge_index
        x, loss_dist = self.complementor(data)
        central_mask = data.central_mask
        if self.cached_edges:
            edge_index1, edge_index2, edge_index = self.edge_index1, self.edge_index2, self.edge_index
        else:
            edge_index1, edge_index2, edge_index = self.graph_partition(data.edge_index, central_mask)
        # print(edge_index1.shape, edge_index2.shape, edge_index.shape)
        
        # adj_sp = torch_sparse.SparseTensor(row=edge_index[1], col=edge_index[0], value=torch.ones(edge_index.shape[1]).to(x.device), sparse_sizes=(x.shape[0], x.shape[0]))
        for ind, conv in enumerate(self.convs):
            # if ind == len(self.convs) -1:
            #     x = conv(x, edge_index, edge_index1, edge_index2, central_mask)
            # else:

            # x = F.elu(conv(x, edge_index, edge_index1, edge_index2, central_mask))
            x = conv(x, edge_index, edge_index1, edge_index2, central_mask)
            if self.use_bn:
                x = self.bns[ind](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
                # x = conv(x, edge_index)
        return x