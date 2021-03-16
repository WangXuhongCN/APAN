import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from torch._C import dtype
from tqdm import tqdm
import csv
import os


def preprocess(args):
  Path("data/").mkdir(parents=True, exist_ok=True)

  if args.data == 'wikipedia' or args.data == 'reddit':
    PATH = ['./data/{}_raw.csv'.format(args.data)]
    feat_dim = 172
  else:
    print('Please check the dataset name.')

  #OUTPUT_GRAPH = f'./data/{args.data}_graph.csv'

  u_list, i_list, ts_list, label_list, idx_list = [], [], [], [], []
  feat_l = []
  Reidx = reindex(args)
  ts_shift = 0
  for data_path in PATH:
    with open(data_path) as dataset:
      s = next(dataset)
      #graph_writer = csv.writer(graph)

      #ndata_writer = csv.writer(ndata)
      #graph_writer.writerow(['u','i','ts','label','idx'])
      for idx, line in enumerate(tqdm(dataset)):
        e = line.strip().split(',')

        u, i = int(e[0]), int(e[1]) 

        u, i = Reidx.user2id(args, u, i)


        ts = float(e[2][:-3]) - ts_shift if args.data == 'alipay' else float(e[2])
        if idx == 0:
          ts_shift = ts
          ts = 0
        if ts == 0:
          ts = 1
        
        label = float(e[3])  # int(e[3])

        feat = [float(x) for x in e[4:]]

        u_list.append(u)
        i_list.append(i)
        ts_list.append(ts)
        label_list.append(label)
        idx_list.append(idx)

        feat_l.append(feat)

        #graph_writer.writerow([u,i,ts,label,idx])
        #edata_writer.writerow(feat)
        #ndata_writer.writerow([0. for n in range(feat_dim)])

  feat = np.array(feat_l, dtype='float32')
  # empty = np.zeros(feat.shape[1], dtype='float32')[np.newaxis, :]
  # feat = np.vstack([empty, feat])


  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), feat

# def write_temporal_subgraph(args, graph, edge_feat):
#   sources = graph.u.values
#   destinations = graph.i.values
#   edge_idxs = graph.idx.values
#   labels = graph.label.values
#   timestamps = graph.ts.values
#   OUTPUT_EDATA = f'./dataset/{args.data}_TSG.csv'

#   max_node_idx = max(sources.max(), destinations.max())
#   adj_list = [[] for _ in range(max_node_idx + 1)]
#   for source, destination, edge_idx, labels, timestamp, edge_feat in tqdm(zip(sources, destinations,
#                                                       edge_idxs, labels,
#                                                       timestamps, edge_feat)):
#     # edge_feat = np.array2string(edge_feat, max_line_width= 99999, precision=8, separator=',', suppress_small=True)
#     # print(edge_feat)
#     edge_feat = edge_feat.tolist()
#     adj_list[source].append([destination, edge_idx, labels, timestamp] + edge_feat)
#     adj_list[destination].append([source, edge_idx, labels, timestamp] + edge_feat)

#   with open(OUTPUT_EDATA,'w') as TSG:
#     csv_writer = csv.writer(TSG)
#     for tsg in tqdm(adj_list):
#       csv_writer.writerow(tsg)


class reindex(object):
  def __init__(self, args):
    super(reindex, self).__init__()
    self.user_idx={}
    self.item_idx={}
    self.curr_idx=0
  def bipartite_graph_reindex(self, u, i):
    if u not in self.user_idx.keys():
      self.user_idx[u] = self.curr_idx
      u = self.curr_idx
      self.curr_idx += 1
    else:
      u=self.user_idx[u]

    if i not in self.item_idx.keys():
      self.item_idx[i] = self.curr_idx
      i = self.curr_idx
      self.curr_idx += 1
    else:
      i=self.item_idx[i]

    return u, i

  def graph_reindex(self, u, i):
    if u not in self.user_idx.keys():
      self.user_idx[u] = self.curr_idx
      u = self.curr_idx
      self.curr_idx += 1
    else:
      u=self.user_idx[u]
    if i not in self.user_idx.keys():
      self.user_idx[i] = self.curr_idx
      i = self.curr_idx
      self.curr_idx += 1
    else:
      i=self.user_idx[i]
    return u, i

  def user2id(self, args, u, i):
    if args.data == 'alipay':
      u, i = self.graph_reindex(u, i)
    else:
      u, i = self.bipartite_graph_reindex(u, i)
    return u, i


parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='alipay')

args = parser.parse_args()

graph, edge_feat = preprocess(args)
OUT_DF = './data/{}.csv'.format(args.data)
OUT_FEAT = './data/{}.npy'.format(args.data)
empty = np.zeros(edge_feat.shape[1], dtype='float32')[np.newaxis, :]
feat = np.vstack([empty, edge_feat])
graph.to_csv(OUT_DF)
np.save(OUT_FEAT, feat)
#write_temporal_subgraph(args, graph, edge_feat)