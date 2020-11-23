import dgl
import torch
import dgl.function as fn
import numpy as np

def get_current_ts(args, pos_graph, neg_graph):
    with pos_graph.local_scope():
        pos_graph_ = dgl.add_reverse_edges(pos_graph, copy_edata=True)
        pos_graph_.update_all(fn.copy_e('timestamp', 'times'), fn.max('times','ts'))  
        current_ts = pos_ts = pos_graph_.ndata['ts']
        num_pos_nodes = pos_graph_.num_nodes()
    if 'LP' in args.tasks:
        with neg_graph.local_scope():
            neg_graph_ = dgl.add_reverse_edges(neg_graph)
            neg_graph_.edata['timestamp'] = pos_graph_.edata['timestamp']
            neg_graph_.update_all(fn.copy_e('timestamp', 'times'), fn.max('times','ts'))
            num_pos_nodes = torch.where(pos_graph_.ndata['ts']>0)[0].shape[0]    
            pos_ts = pos_graph_.ndata['ts'][:num_pos_nodes]
            neg_ts = neg_graph_.ndata['ts'][num_pos_nodes:]
            current_ts = torch.cat([pos_ts,neg_ts])
    return current_ts, pos_ts, num_pos_nodes

def set_random_seeds(seed):
    if seed != -1:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dgl.seed(seed)
        # torch.backends.cudnn.deterministic=True