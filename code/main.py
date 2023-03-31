# -*- coding: utf-8 -*-
import pickle
import time
import argparse
import os
import logging
from model import *
from utils import handle_adj_weight
from Datasets import SessionData

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='sample/Tmall/yoochoose/Nowplaying')
parser.add_argument('--hidden_size', type=int, default=100, help='the size of hidden layers')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to train for')
parser.add_argument('--sample_num', type=int, default=12)
parser.add_argument('--GNN', default='GGNN', help='GGNN/GCN/GAT')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--rate', type=float, default=0.2, help='GCL loss rate')
parser.add_argument('--tua', type=float, default=0.05, help='tua about GCL')
parser.add_argument('--eps', type=float, default=0.10, help='eps about GCL')  # 0.1 0.2
parser.add_argument('--dropout_nei', type=float, default=0.5, help='Dropout rate of neighbor pool.')
parser.add_argument('--dropout_gnn', type=float, default=0.5, help='Dropout rate of GNN.')
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate of global preference')
parser.add_argument('--dropout_session', type=float, default=0.5, help='Dropout rate of short preference.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--top_k', type=int, default=20, help='top K indicator for evaluation')
parser.add_argument('--seed', type=int, default=None, help='Seed number.')
opt = parser.parse_args()
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def init_seed(logger, seed=None):
    if seed is None:
        seed = int((time.time() * 1000) % 10000)
    logger.info('Random seed: {}'.format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_creator():
    cur_dir = os.getcwd()
    output_dir = cur_dir + '/logs/' + str(opt.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%m-%d-%H') +
                               str([opt.dataset, opt.rate, opt.tua, opt.eps,
                                    opt.dropout_nei, opt.dropout_gnn, opt.dropout_global, opt.dropout_session]))
    final_log_file = os.path.join(output_dir, log_name)

    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")

    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log


def main():
    if opt.dataset == "Tmall":
        num_node = 40727
        opt.rate = 0.30
        opt.dropout_nei = 0.60
        opt.dropout_gnn = 0.20
        opt.dropout_global = 0.60  # 0.5~0.6
        opt.dropout_session = 0.00
    elif opt.dataset == 'yoochoose':
        num_node = 37483
        opt.rate = 0.15  # 0.20
        opt.dropout_nei = 0.80
        opt.dropout_gnn = 0.60  # 0.2 or 0.6
        opt.dropout_global = 0.20
        opt.dropout_session = 0.60
    elif opt.dataset == "Nowplaying":
        num_node = 60417
        opt.rate = 0.15
        opt.tua = 0.07
        opt.eps = 0.20
        opt.dropout_nei = 0.80  # 0.20
        opt.dropout_gnn = 0.20  # 0.00
    else:
        num_node = 309

    logger = log_creator()
    init_seed(logger, seed=opt.seed)
    cur_dir = os.getcwd()

    adj = pickle.load(open(os.path.join('datasets', opt.dataset, 'adj_{}.pkl'.format(opt.sample_num)), 'rb'))
    weight = pickle.load(open(os.path.join('datasets', opt.dataset, 'weight_{}.pkl'.format(opt.sample_num)), 'rb'))
    train_data = SessionData(os.path.join(cur_dir, 'datasets', opt.dataset), phrase='train')
    test_data = SessionData(os.path.join(cur_dir, 'datasets', opt.dataset), phrase='test')
    adj, weight = handle_adj_weight(adj, weight, num_node, opt.sample_num)

    model = trans_to_cuda(RNGCL(opt, num_node, adj, weight))  # init model

    logger.info(opt)
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        logger.info('-'*50)
        logger.info('epoch: ' + str(epoch))
        hit, mrr = train_test(model, train_data, test_data, logger, top_k=opt.top_k, rate=opt.rate)
        flag = 0
        if hit > best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr > best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1

        logger.info('Current Result:')
        logger.info('\tRecall@20:\t%.4f\tMRR@20:\t%.4f' % (hit, mrr))
        logger.info('Best Result:')
        logger.info('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
                    best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break

    logger.info('-'*50)
    end = time.time()
    model_dir = './model_params/'+opt.GNN
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model, model_dir + '/' + opt.dataset + str([float(format(x, '.5')) for x in best_result]) + '.pt')
    logger.info("Run time: %f min" % ((end - start)/60))


if __name__ == '__main__':
    main()
