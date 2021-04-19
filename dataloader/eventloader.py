import dgl
import torch
import numpy as np
import random

class TemporalEdgeCollator(dgl.dataloading.EdgeCollator):
    def __init__(self, args, g, eids, block_sampler, g_sampling=None, exclude=None,
                 reverse_eids=None, reverse_etypes=None, negative_sampler=None, mode='val'):
        super(TemporalEdgeCollator, self).__init__(g, eids, block_sampler, g_sampling, exclude,
                 reverse_eids, reverse_etypes, negative_sampler)
        
        self.args = args
        self.mode = mode

    def collate(self, items):
        #print('before', self.block_sampler.ts)

        current_ts = self.g.edata['timestamp'][items[-1]]  # only sample edges before last timestamp in a batch
        self.block_sampler.ts = current_ts
        neg_pair_graph = None
        if self.negative_sampler is None:
            input_nodes, pair_graph, blocks = self._collate(items)
        else:
            input_nodes, pair_graph, neg_pair_graph, blocks = self._collate_with_negative_sampling(items)

        for i in range(self.args.n_layer-1):
            self.block_sampler.frontiers[0].add_edges(*self.block_sampler.frontiers[i+1].edges())
        frontier = dgl.reverse(self.block_sampler.frontiers[0])

        return input_nodes, pair_graph, neg_pair_graph, blocks, frontier, current_ts

class MultiLayerTemporalNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, args, fanouts, replace=False, return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts
        self.replace = replace
        self.ts = 0
        self.args = args
        self.frontiers = [None for _ in range(len(fanouts))]

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]

        g = dgl.in_subgraph(g, seed_nodes)
        g.remove_edges(torch.where(g.edata['timestamp']>self.ts)[0])

        if fanout is None:
            frontier = g
            #frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            if self.args.uniform:
                frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
            else:
                frontier = dgl.sampling.select_topk(g, fanout, 'timestamp', seed_nodes)

        self.frontiers[block_id] = frontier
        return frontier

class frauder_sampler():
    def __init__(self, g):
        self.fraud_eid = torch.where(g.edata['label']!=0)[0]
        len_frauder = self.fraud_eid.shape[0] // 2
        self.fraud_eid = self.fraud_eid[:len_frauder]
        self.ts = g.edata['timestamp'][self.fraud_eid]
    def sample_fraud_event(self, g, bs, current_ts):
        idx = (self.ts<current_ts)
        num_fraud = idx.sum().item()
        
        if num_fraud > bs:
            
            idx[random.sample(list(range(num_fraud)), num_fraud-bs)] = False # 只采样一部分fraud event
            
        fraud_eid = self.fraud_eid[idx]
        
        fraud_graph = dgl.edge_subgraph(g, fraud_eid)
        return fraud_graph
