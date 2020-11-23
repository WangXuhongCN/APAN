import torch
from torch import nn
import dgl
import dgl.function as fn

class Msg2Mail():
    def __init__(self, args, nfeat_dim):
        super(Msg2Mail, self).__init__()
        self.args = args
        self.nfeat_dim = nfeat_dim
        if args.edgedrop:
            self.edgedrop = nn.Dropout(args.edgedrop)
        else:
            self.edgedrop = 0.       
    def gen_mail(self, args, emb, input_nodes, pair_graph, frontier, mode='train'):
        pair_graph.ndata['feat'] = emb
        # pair_graph = pair_graph.to('cpu')
        pair_graph = dgl.add_reverse_edges(pair_graph, copy_edata=True)
        # TODO:下面这里用mean总觉得不太对，用sum会梯度爆炸，看来还是要自定义才行
        # 把mailbox里的东西统统都放到mail里面，而不是先mean
        pair_graph.update_all(MSG.get_edge_msg, fn.mean('m','msg')) 
        frontier.ndata['msg'] = torch.zeros((frontier.num_nodes(), self.nfeat_dim + 2))
        frontier.ndata['msg'][pair_graph.ndata[dgl.NID]] = pair_graph.ndata['msg'].to('cpu')
        #frontier.ndata['pv'] = compute_pagerank(frontier)
        for _ in range(args.n_layer):
            # TODO: 这里可以加edge dropout https://github.com/dmlc/dgl/blob/f99725adbc9cef2fd6cc6eef18a86e3d7c1e5339/examples/pytorch/appnp/appnp.py
            if self.args.edgedrop and mode == 'train':
                # performing edge dropout
                ed = self.edgedrop(torch.ones((frontier.number_of_edges(), 1))).bool().float()
                frontier.edata['d'] = ed
                frontier.update_all(fn.src_mul_edge(src='msg', edge='d', out='m'),
                                fn.mean(msg='m', out='msg'))
            else:
                frontier.update_all(fn.copy_u('msg','m'), fn.mean('m','msg'))
            # 这里不应该全局update，仅仅以某些点为出发点发送消息即可
            # 把mailbox里的东西统统都放到mail里面，而不是先mean
            
            #frontier.update_all(MSG.pass_msg, fn.mean('m','msg'))
        mail = MSG.msg2mail(frontier.ndata['mail'][input_nodes], frontier.ndata['msg'][input_nodes])
        return mail


def compute_pagerank(g, DAMP = 0.85, K = 10):
    N = g.num_nodes()
    g.ndata['pv'] = torch.ones(N) / N
    degrees = g.out_degrees(g.nodes()).type(torch.float32)
    for k in range(K):
        g.ndata['pv'] = g.ndata['pv'] / degrees
        g.update_all(message_func=fn.copy_src(src='pv', out='m'),
                     reduce_func=fn.sum(msg='m', out='pv'))
        g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['pv']
    return g.ndata['pv']


class MSG():
    def __init__(self):
        super(MSG, self).__init__()
    def get_edge_msg(edges):
        msg = edges.src['feat'] + edges.dst['feat'] + edges.data['feat']
        loc_info = torch.tensor(1).repeat(edges.batch_size()).unsqueeze(1).float()
        if msg.is_cuda:
            device = msg.device
            loc_info = loc_info.to(device)
        msg = torch.cat((msg, edges.data['timestamp'].unsqueeze(1), loc_info), 1)
        return {'m': msg}
    def pass_msg(edges):
        loc_info = torch.tensor(1).repeat(edges.batch_size()).unsqueeze(1)
        msg = torch.cat((edges.src['msg'], edges.src['ts'].unsqueeze(1), loc_info), 1)
        # ppr = edges.src['pv']
        # # edge_ts = edges.data['timestamp']
        # # msg[:,-2] = msg[:,-2] - edge_ts
        # msg[:,-1] = ppr
        #msg = edges.src['msg']* 0.5
        #print(msg)
        return {'m': msg}
    def msg2mail(mail, msg):

        mail = torch.cat((msg.unsqueeze(1), mail[:, :-1]), 1)
        return mail
    # def msg_reduce(nodes):
    #     msg = nodes.mailbox['m']
    #     mail = torch.cat((msg.unsqueeze(1), nodes.data['mail'][:, :-1]), 1)
    #     return {'mail': mail}
    # def msg_reduce(nodes):
    #     msg = nodes.mailbox['m'].sum(1)
    #     mail = torch.cat((msg.unsqueeze(1), nodes.data['mail'][:, :-1]), 1)
    #     return {'mail': mail}
    # def msg2mail(nodes):
    #     msg = nodes.mailbox['m']
    #     print(msg.shape)
    #     n_msg = msg.shape[1]
    #     print(n_msg)
    #     #mail = torch.cat((msg.unsqueeze(1), nodes.data['mail'][:, :-n_msg]), 1)
    #     #mail = torch.cat((msg.unsqueeze(1), nodes.data['mail'][:, :-n_msg]), 1)
    #     return {'mail': msg}
# 传播的时候要距离编码，时间编码，这个编码最好是不需要学习的