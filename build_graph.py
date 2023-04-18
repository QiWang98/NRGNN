# -*- coding: utf-8 -*-
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='sample/Tmall/yoochoose/Nowplaying')
parser.add_argument('--sample_num', type=int, default=12)
parser.add_argument('--k_hop', type=int, default=1)
opt = parser.parse_args()

sample_num = opt.sample_num

seq = pickle.load(open('datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))

if opt.dataset == "Tmall":
    num = 40727
elif opt.dataset == 'yoochoose':
    num = 37483
elif opt.dataset == "Nowplaying":
    num = 60417
else:
    num = 309

relation = []
neighbor = [] * num

all_test = set()

adj1 = [dict() for _ in range(num)]
adj = [[] for _ in range(num)]

for i in range(len(seq)):
    data = seq[i]
    for k in range(1, opt.k_hop+1):
        for j in range(len(data)-k):
            relation.append([data[j]-1, data[j+k]-1])
            relation.append([data[j+k]-1, data[j]-1])  # undirected graph

for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1

weight = [[] for _ in range(num)]

for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]

for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

pickle.dump(adj, open('datasets/' + opt.dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('datasets/' + opt.dataset + '/weight_' + str(sample_num) + '.pkl', 'wb'))
print("Build graph of %s" % opt.dataset)
