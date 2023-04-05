# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, GCNConv, GATConv


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def handle_adj_weight(adj_dict, weight_dict, num_node, sample_num):
    adj_entity = np.zeros([num_node, sample_num], dtype=np.int64)
    weight_entity = np.zeros([num_node, sample_num], dtype=np.float32)
    for i in range(0, num_node):
        neighbor = adj_dict[i]
        neighbor_weight = weight_dict[i]
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            adj_entity[i] = np.full(sample_num, i)
            weight_dict[i] = np.ones(sample_num, dtype=np.float)
            continue
        elif n_neighbor < sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num - n_neighbor, replace=True)
            adj_entity[i] = np.concatenate((adj_dict[i], np.array([neighbor[i] for i in sampled_indices])))
            weight_entity[i] = np.concatenate((weight_dict[i], np.array([neighbor_weight[i] for i in sampled_indices])))
        else:
            adj_entity[i] = neighbor
            weight_entity[i] = neighbor_weight

    return torch.from_numpy(adj_entity).long(), F.normalize(torch.from_numpy(weight_entity))


def translate_to_seq(session_embed, alias_session):
    seq_hidden = trans_to_cuda(torch.zeros(alias_session.shape[0], alias_session.shape[1], session_embed[0].shape[-1]))
    for i in range(alias_session.shape[0]):
        seq_hidden[i] = session_embed[i][alias_session[i]]

    return seq_hidden


class SessionGraphCL(nn.Module):
    def __init__(self, opt, hidden_size):
        super(SessionGraphCL, self).__init__()
        self.opt = opt
        self.hidden_size = hidden_size
        if self.opt.GNN == "GGNN":
            self.ggnn = GatedGraphConv(self.hidden_size, num_layers=1)
        elif self.opt.GNN == "GCN":
            self.gcn = GCNConv(self.hidden_size, self.hidden_size)
        elif self.opt.GNN == "GAT":
            self.gat_1 = GATConv(self.hidden_size, self.hidden_size, heads=4)
            self.gat_2 = GATConv(self.hidden_size * 4, self.hidden_size)
        else:
            raise "No GNN is specified"
        self.w1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=self.opt.alpha)

    def InfoNCE(self, view1, view2):
        temperature = self.opt.tua
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def forward(self, x_embed, data):
        edge_index, batch = data.edge_index, data.batch
        mask = data.mask.unsqueeze(-1)
        """
        # drop_edges
        edge_num = edge_index.shape[-1]
        edge_mask = torch.rand((edge_num,)) >= int(edge_num * self.opt.drop_edges) / edge_num
        edge_index_cl = edge_index[:, edge_mask]
        node_embed = self.leaky_relu(self.gcn(x_embed, edge_index)) + x_embed
        node_embed_cl = self.leaky_relu(self.gcn(x_embed, edge_index_cl)) + x_embed
        # drop_nodes
        num_nodes = data.num_nodes
        num_remove_nodes = int(num_nodes * self.opt.drop_nodes)
        remove_nodes = trans_to_cuda(torch.randint(0, num_nodes, (num_remove_nodes,)))
        nodes_mask = torch.logical_or(torch.isin(edge_index[0], remove_nodes),
                                      torch.isin(edge_index[1], remove_nodes))
        edge_index_cl = edge_index[:, ~nodes_mask]
        node_embed = self.leaky_relu(self.gcn(x_embed, edge_index)) + x_embed
        node_embed_cl = self.leaky_relu(self.gcn(x_embed, edge_index_cl)) + x_embed
        """
        random_noise = torch.randn_like(x_embed)
        x_embed_cl = x_embed + F.normalize(random_noise, dim=-1) * self.opt.eps
        if self.opt.GNN == "GGNN":
            node_embed = self.ggnn(x_embed, edge_index) + x_embed
            node_embed_cl = self.ggnn(x_embed_cl, edge_index) + x_embed_cl
        elif self.opt.GNN == "GCN":
            node_embed = self.leaky_relu(self.gcn(x_embed, edge_index)) + x_embed
            node_embed_cl = self.leaky_relu(self.gcn(x_embed_cl, edge_index)) + x_embed_cl
        elif self.opt.GNN == "GAT":
            node_embed = self.leaky_relu(self.gat_1(x_embed, edge_index))
            node_embed_cl = self.leaky_relu(self.gat_1(x_embed_cl, edge_index))
            node_embed = self.gat_2(node_embed, edge_index) + x_embed
            node_embed_cl = self.gat_2(node_embed_cl, edge_index) + x_embed_cl
        else:
            raise "No GNN is specified"
        gcl_loss = self.InfoNCE(node_embed, node_embed_cl)

        section = torch.bincount(data.batch)
        session_embed = torch.split(node_embed, list(section.cpu().numpy()))
        seq_embed = translate_to_seq(session_embed, data.alias_session) * mask.float()
        session_mean_embed = F.relu(self.w2(torch.mean(seq_embed, dim=1))).unsqueeze(dim=1).repeat(1, mask.shape[1], 1)
        session_mean_embed = F.dropout(session_mean_embed, p=self.opt.dropout_gnn, training=self.training)
        session_pre = seq_embed + session_mean_embed

        return session_pre, gcl_loss


class NeighborPooling(nn.Module):
    def __init__(self, opt, hidden_size, embedding):
        super(NeighborPooling, self).__init__()
        self.opt = opt
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.w_1 = nn.Linear(self.hidden_size * 2 + 1, self.hidden_size)
        self.w_2 = nn.Linear(self.hidden_size, 1, bias=False)
        self.w_3 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=opt.alpha)

    def forward(self, x_embed, data, adj, weight):
        mask = data.mask.unsqueeze(-1).float()
        seq_lens = mask.shape[1]
        all_neighbor_adj = adj[data.session]
        all_neighbor_weight = weight[data.session]

        section = torch.bincount(data.batch)
        session_embed = torch.split(x_embed, list(section.cpu().numpy()))
        session_embed = translate_to_seq(session_embed, data.alias_session) * mask.float()
        session_mean_embed = session_embed.sum(dim=1) / mask.sum(dim=1)

        sample_num = all_neighbor_adj.shape[-1]
        all_neighbor_embedding = self.embedding(all_neighbor_adj)
        alpha = self.w_1(torch.cat([session_mean_embed.unsqueeze(dim=1).unsqueeze(1).repeat(1, seq_lens, sample_num, 1)
                                    , all_neighbor_embedding, all_neighbor_weight.unsqueeze(-1)], dim=-1))
        alpha = self.w_2(self.leaky_relu(alpha)).squeeze(-1)
        alpha = F.softmax(alpha, dim=-1).unsqueeze(-1)
        neighbor_vector = torch.sum(alpha * all_neighbor_embedding, dim=2)
        last_item_pre = session_embed[:, 0, :] + neighbor_vector[:, 0, :]

        neighbor_vector = F.dropout(neighbor_vector, p=self.opt.dropout_nei, training=self.training)
        global_pre = session_embed + neighbor_vector

        return global_pre, last_item_pre
