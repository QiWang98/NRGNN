# -*- coding: utf-8 -*-
import datetime
import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from utils import trans_to_cuda, trans_to_cpu, NeighborPooling, SessionGraphCL


class RNGCL(nn.Module):
    def __init__(self, opt, num_node, adj_all, weight_all):
        super(RNGCL, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.hidden_size = opt.hidden_size
        self.num_node = num_node
        self.adj_all = trans_to_cuda(adj_all)
        self.weight_all = trans_to_cuda(weight_all)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.hidden_size)
        self.pos_embedding = nn.Embedding(200, self.hidden_size)

        # Aggregator
        self.nei_pool = NeighborPooling(self.opt, self.hidden_size, self.embedding)
        self.session_gcl = SessionGraphCL(self.opt, self.hidden_size)

        # Parameters
        self.w1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.glu1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=opt.alpha)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=opt.lr_dc_step)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        mask = data.mask.unsqueeze(-1)
        batch_size = mask.shape[0]
        seq_lens = mask.shape[1]
        x_embed = self.embedding(data.x).squeeze()

        global_pre, last_item_pre = self.nei_pool(x_embed, data, self.adj_all, self.weight_all)
        session_pre, gcl_loss = self.session_gcl(x_embed, data)

        global_pre = F.dropout(global_pre, p=self.opt.dropout_global, training=self.training)
        session_pre = F.dropout(session_pre, p=self.opt.dropout_session, training=self.training)
        total_pre = (global_pre + session_pre) * mask.float()

        pos_embed = self.pos_embedding.weight[:seq_lens]
        pos_embed = pos_embed.unsqueeze(0).repeat(batch_size, 1, 1)

        session_hidden = torch.sum(total_pre * mask, dim=1) / torch.sum(mask, dim=1)
        session_hidden = session_hidden.unsqueeze(1).repeat(1, seq_lens, 1)

        seq_pos_hidden = torch.tanh(self.w1(pos_embed + total_pre))

        beta = self.w2(torch.sigmoid(self.glu1(seq_pos_hidden) + self.glu2(session_hidden)))
        beta = beta * mask.float()

        hidden = torch.sum(beta * total_pre, dim=1) + last_item_pre

        init_embedding = self.embedding.weight
        scores = torch.matmul(hidden, init_embedding.transpose(1, 0))

        return scores, gcl_loss


def train_test(model, train_data, test_data, logger, top_k, rate):
    logger.info('start training: ' + str(datetime.datetime.now()))
    model.train()
    train_loader = DataLoader(train_data, num_workers=1, batch_size=model.batch_size, shuffle=True, pin_memory=True)
    train_rc_loss_list, train_cl_loss_list = [], []

    for i, data in enumerate(tqdm(train_loader)):
        data = trans_to_cuda(data)
        model.optimizer.zero_grad()
        scores, gcl_loss = model(data)
        targets = data.y
        loss = model.loss_function(scores, targets) * (1-rate) + gcl_loss * rate
        loss.backward()
        model.optimizer.step()
        train_rc_loss_list.append((loss - gcl_loss * rate) / (1 - rate))
        train_cl_loss_list.append(gcl_loss)
    train_rc_loss = (sum(train_rc_loss_list)/len(train_rc_loss_list)).item()
    train_gcl_loss = (sum(train_cl_loss_list)/len(train_cl_loss_list)).item()
    logger.info('\tTrain Rec Loss:\t%.3f' % train_rc_loss)
    logger.info('\tTrain GCL Loss:\t%.3f' % train_gcl_loss)

    model.scheduler.step()

    logger.info('start predicting: ' + str(datetime.datetime.now()))
    model.eval()
    test_loader = DataLoader(test_data, num_workers=1, batch_size=model.batch_size, shuffle=True, pin_memory=True)
    result = []
    hit, mrr = [], []
    test_rc_loss_list, test_gcl_loss_list = [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            data = trans_to_cuda(data)
            scores, gcl_loss = model(data)
            targets = data.y
            sub_scores = scores.topk(top_k)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            targets = targets.cpu().numpy()

            test_rc_loss_list.append(model.loss_function(scores, trans_to_cuda(torch.from_numpy(targets))))
            test_gcl_loss_list.append(gcl_loss)

            for score, target, mask in zip(sub_scores, targets, data.mask):
                hit.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target)[0][0] + 1))

        test_rc_loss = (sum(test_rc_loss_list)/len(test_rc_loss_list)).item()
        test_gcl_loss = (sum(test_gcl_loss_list)/len(test_gcl_loss_list)).item()
        logger.info('\tTest Rec Loss:\t%.3f' % test_rc_loss)
        logger.info('\tTest GCL Loss:\t%.3f' % test_gcl_loss)

        result.append(np.mean(hit) * 100)
        result.append(np.mean(mrr) * 100)

    return result
