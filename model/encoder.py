import torch
from torch import nn
import numpy as np
import dgl.function as fn

class Encoder(nn.Module):
    def __init__(self, args, emb_dim, n_head=2, dropout=0.1, use_mask=False):
        super(Encoder, self).__init__()
        self.args = args

        self.attn_aggr = mail_attn_agger(emb_dim, n_head, dropout)
        self.time_encoder = TimeEncode(emb_dim)
        len_mail = args.n_mail
        self.pos_encoder = PosEncode(emb_dim, len_mail)
        self.merger = MergeLayer((emb_dim)*2, emb_dim, dropout=dropout)
        self.use_mask = use_mask

    def forward(self, pos_graph, neg_graph, num_pos_nodes):


        mail = pos_graph.ndata['mail']

        mask = None
        if self.use_mask:
            mask = (mail.mean(2)==0).bool()
            mask[:,0] = 0
            
        feat = pos_graph.ndata['feat']

        joined_mail = mail[:,:,:-2]

        if not self.args.no_time: 
            ts = pos_graph.ndata['ts']
            last_update = pos_graph.ndata['last_update']
            mail_time = mail[:,:,-2]
            delta_t_msg = ts.unsqueeze(1) - mail_time
            time_emb_msg = self.time_encoder(delta_t_msg)
            joined_mail = joined_mail + time_emb_msg
            delta_t_feat =  ts.unsqueeze(1) - last_update.unsqueeze(1)
            time_emb_feat = self.time_encoder(delta_t_feat)
            feat = feat + time_emb_feat.squeeze()
        if not self.args.no_pos:
            mail_time = mail[:,:,-2]
            pos_emb_msg = self.pos_encoder(mail_time)
            joined_mail = joined_mail + pos_emb_msg
        
        attn_output, attn_weight = self.attn_aggr(feat, joined_mail, mask)
        attn_output = self.merger(feat, attn_output)
        return attn_output, attn_weight

def mails_cat(nodes):
    return {'cat_mail': torch.cat([nodes.data['mail'], nodes.data['opposite_mail']],1)}

class mail_attn_agger(nn.Module):
    def __init__(self, emb_dim, n_head=2, dropout=0.1):
        super(mail_attn_agger, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim,
                                                    kdim=emb_dim,
                                                    vdim=emb_dim,
                                                    num_heads=n_head,
                                                    dropout=dropout)
        
    # 距离编码、时间编码
    def forward(self, feat, mail, mask):
        mail = mail.permute([1, 0, 2])
        attn_output, attn_weight = self.multihead_attn(feat.unsqueeze(0), mail, mail, mask)
        attn_output, attn_weight = attn_output.squeeze(), attn_weight.squeeze()

        return attn_output, attn_weight

class TimeEncode(torch.nn.Module):

    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                        .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())


    def forward(self, t):
        # t has shape [batch_size, seq_len]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=2)
        # output has shape [batch_size, seq_len, dimension]
        t = self.w(t)
        output = torch.cos(t)
        return output

class PosEncode(nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()
        
        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)
        
    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb

class MergeLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(in_dim)
        hidden_dim = (in_dim+out_dim)//2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.relu = nn.ReLU()

        # torch.nn.init.xavier_normal_(self.fc1.weight)
        # torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        h = torch.cat([x1, x2], dim=1)
        h = self.norm(h)
        h = self.dropout(h)
        #h = self.norm(h)
        h = self.fc1(h)
        h = self.relu(h)
        #h = self.dropout(h)
        h = self.fc2(h)
        
        return h 