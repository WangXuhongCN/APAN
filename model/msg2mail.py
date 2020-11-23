import torch
from torch import nn
import dgl
import dgl.function as fn

class Msg2Mail():
    def __init__(self, args, nfeat_dim):
        super(Msg2Mail, self).__init__()
        self.args = args
        self.nfeat_dim = nfeat_dim
     
    def gen_mail(self, args, emb, input_nodes, pair_graph, frontier, mode='train'):
        pair_graph.ndata['feat'] = emb

        pair_graph = dgl.add_reverse_edges(pair_graph, copy_edata=True)

        pair_graph.update_all(MSG.get_edge_msg, fn.mean('m','msg')) 
        frontier.ndata['msg'] = torch.zeros((frontier.num_nodes(), self.nfeat_dim + 2))
        frontier.ndata['msg'][pair_graph.ndata[dgl.NID]] = pair_graph.ndata['msg'].to('cpu')

        for _ in range(args.n_layer):
            frontier.update_all(fn.copy_u('msg','m'), fn.mean('m','msg'))

        mail = MSG.msg2mail(frontier.ndata['mail'][input_nodes], frontier.ndata['msg'][input_nodes])
        return mail


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
        return {'m': msg}
    def msg2mail(mail, msg):

        mail = torch.cat((msg.unsqueeze(1), mail[:, :-1]), 1)
        return mail
