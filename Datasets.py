# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import pickle
from torch_geometric.data import InMemoryDataset, Data
import os


def handle_data(input_data, train_len=None):
    seq_length = [len(cur_data) for cur_data in input_data]
    if train_len is None:
        max_len = max(seq_length)
    else:
        max_len = train_len

    # reverse the sequence
    data_eq_seq = [np.concatenate([np.asarray(list(reversed(seq)))-1, np.zeros(max_len-le, dtype=int)])
                   if le < max_len else np.asarray(list(reversed(seq[-max_len:])))
                   for seq, le in zip(input_data, seq_length)]
    data_masks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
                  for le in seq_length]

    return data_eq_seq, data_masks


class SessionData(InMemoryDataset):
    def __init__(self, raw_dir, phrase, session_len=None):
        assert phrase in ['train', 'test']
        self.phrase = phrase
        self.session_len = session_len
        super(SessionData, self).__init__(raw_dir)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.phrase + '.txt']

    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']

    def download(self):
        pass

    def process(self):
        data = pickle.load(open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'rb'))

        sessions, masks = handle_data(data[0], self.session_len)
        sessions = np.asarray(sessions)
        targets = torch.LongTensor(data[1]) - 1
        masks = torch.tensor(masks)
        graph_list = []
        for session, target, mask in zip(sessions, targets, masks):
            u_list = []
            v_list = []
            edge_dict = {}
            num_nodes = mask.sum().item()
            nodes = pd.unique(pd.Series(session[0:num_nodes]))  # return array

            for i in range(len(session)-1):
                v = np.where(nodes == session[i])[0][0]  # due to the reversed session, u and v are reversed
                if mask[i+1] == 0:
                    break
                u = np.where(nodes == session[i+1])[0][0]
                u_list.append(u)
                v_list.append(v)

            for u_v in zip(u_list, v_list):
                if u_v not in edge_dict:
                    edge_dict[u_v] = 1
                else:
                    edge_dict[u_v] += 1

            edge_index = [[], []]
            for index, weight in edge_dict.items():
                edge_index[0].append(index[0])
                edge_index[1].append(index[1])

            alias_session = [np.where(nodes == session[i])[0][0] if mask[i] != 0 else 0 for i in range(len(session))]

            nodes = torch.LongTensor(nodes).reshape(-1, 1)
            edge_index = torch.LongTensor(edge_index)
            session = torch.LongTensor(session).unsqueeze(0)
            alias_session = torch.LongTensor(alias_session).unsqueeze(0)
            mask = mask.unsqueeze(0)

            session_graph = Data(x=nodes, y=target, edge_index=edge_index, session=session,
                                 alias_session=alias_session, mask=mask)
            graph_list.append(session_graph)

        data, slices = self.collate(graph_list)
        torch.save((data, slices), self.processed_paths[0])
