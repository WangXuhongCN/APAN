import dgl
import torch
import argparse
import pandas as pd
import numpy as np
from dgl.data.utils import save_graphs

import dgl.function as fn

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('-d', '--data', type=str, choices=["wikipedia", "reddit", "alipay"], help='Dataset name (eg. wikipedia or reddit)',
                        default='alipay')
parser.add_argument('--new_node_count', action='store_true',
                        help='count how many nodes are not in training set')    
args = parser.parse_args()
args.new_node_count = True
# if args.data == 'alipay':
#     feat_dim = 101
# else:
#     feat_dim = 172

graph_df = pd.read_csv('./data/{}.csv'.format(args.data))
edge_features = np.load('./data/{}.npy'.format(args.data))
nfeat_dim = edge_features.shape[1]


src = torch.tensor(graph_df.u.values)
dst = torch.tensor(graph_df.i.values)
label = torch.tensor(graph_df.label.values, dtype=torch.float32)
timestamp = torch.tensor(graph_df.ts.values, dtype=torch.float32)
edge_feat = torch.tensor(edge_features[1:], dtype=torch.float32)
if args.data == 'alipay':
    nfeat_dim += 3 # 为了确保被多头注意力的头数整除
    edge_feat = torch.cat([edge_feat, torch.zeros((edge_feat.shape[0],3))],1)
g = dgl.graph((torch.cat([src,dst]), torch.cat([dst,src])))
len_event = src.shape[0]

g.edata['label'] = label.repeat(2).squeeze()
g.edata['timestamp'] = timestamp.repeat(2).squeeze()
g.edata['feat'] = edge_feat.repeat(2,1).squeeze()
if args.data == 'alipay':
    print(g)
    print('fraud percent：', torch.where(g.edata['label'] != 0)[0].shape[0]/g.num_edges())
    low_degree_nodes = torch.where(g.in_degrees() <= 2)[0]
    del_in_edges = g.in_edges(low_degree_nodes)
    del_out_edges = g.out_edges(low_degree_nodes)
    del_eid = torch.cat([g.edge_ids(*del_in_edges), g.edge_ids(*del_out_edges)])
    g.remove_edges(del_eid)
    
    g = dgl.compact_graphs(g)
    # fraud_edges = g.find_edges(torch.where(g.edata['label'] != 0)[0])
    # # related_nodes = torch.cat
    # g.remove_nodes(torch.where(g.in_degrees() <= 2)[0])
    print('fraud percent：', torch.where(g.edata['label'] != 0)[0].shape[0]/g.num_edges())
    del g.ndata['_ID']
    del g.edata['_ID']
print(g)
#save_graphs(f"./data/{args.data}.dgl", g)

if args.new_node_count:
    origin_num_edges = g.num_edges()//2
    train_eid = torch.arange(0, int(0.7 * origin_num_edges))
    un_train_eid = torch.arange(int(0.7 * origin_num_edges), origin_num_edges)

    train_g = dgl.graph(g.find_edges(train_eid))
    val_n_test_g = dgl.compact_graphs(dgl.graph(g.find_edges(un_train_eid)))

    print(f'total nodes: {g.num_nodes()}, training nodes: {train_g.num_nodes()}, val_n_test nodes: {val_n_test_g.num_nodes()}')
    old_nodes = val_n_test_g.num_nodes()-g.num_nodes()+train_g.num_nodes()
    print(f'old nodes in val_n_test: {old_nodes} ({round((old_nodes)*100/val_n_test_g.num_nodes(),4)}%)')
    new_nodes = g.num_nodes()-train_g.num_nodes()
    print(f'new nodes in val_n_test: {new_nodes} ({round((new_nodes)*100/val_n_test_g.num_nodes(),4)}%)')