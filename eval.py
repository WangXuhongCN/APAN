from pytorch_lightning.metrics.functional import accuracy, auroc, average_precision, roc, f1
import torch
from model import Msg2Mail
import numpy as np
import dgl
import time
# from args import get_args
from utils import get_current_ts, get_args

def eval_epoch(args, logger, g, dataloader, encoder, decoder, msg2mail, loss_fcn, device, num_samples):

    m_ap, m_auc, m_acc = [[], [], []] if 'LP' in args.tasks else [0, 0, 0]

    labels_all = torch.zeros((num_samples)).long()
    logits_all = torch.zeros((num_samples))
    
    attn_weight_all = torch.zeros((num_samples, args.n_mail))

    m_loss = []
    m_infer_time = []
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        loss = torch.tensor(0)
        for batch_idx, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(dataloader):
            n_sample = pos_graph.num_edges()
            start_idx = batch_idx * n_sample
            end_idx = min(num_samples, start_idx + n_sample)
            
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device) if neg_graph is not None else None
            if not args.no_time or not args.no_pos:
                current_ts, pos_ts, num_pos_nodes = get_current_ts(args, pos_graph, neg_graph)
                pos_graph.ndata['ts'] = current_ts
            else:
                current_ts, pos_ts, num_pos_nodes = None, None, None

            _ = dgl.add_reverse_edges(neg_graph) if neg_graph is not None else None

            start = time.time()
            emb, attn_weight = encoder(dgl.add_reverse_edges(pos_graph), _, num_pos_nodes)
            #attn_weight_all[start_idx:end_idx] = attn_weight[:n_sample]
           
            logits, labels = decoder(emb, pos_graph, neg_graph)
            end = time.time() - start
            m_infer_time.append(end)

            loss = loss_fcn(logits, labels)
            m_loss.append(loss.item())
            mail = msg2mail.gen_mail(args, emb, input_nodes, pos_graph, frontier, 'val')
            if not args.no_time:
                g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts.to('cpu')
            g.ndata['feat'][pos_graph.ndata[dgl.NID]] = emb.to('cpu')
            g.ndata['mail'][input_nodes] = mail            

            labels = labels.long()
            logits = logits.sigmoid()
            if 'LP' in args.tasks:
                pred = logits > 0.5
                m_ap.append(average_precision(logits, labels).cpu().numpy())
                m_auc.append(auroc(logits, labels).cpu().numpy())
                m_acc.append(accuracy(pred, labels).cpu().numpy())
            else:
                labels_all[start_idx:end_idx] = labels
                logits_all[start_idx:end_idx] = logits
            
    if 'LP' in args.tasks:
        ap, auc, acc = np.mean(m_ap), np.mean(m_auc), np.mean(m_acc)
    else:
        pred_all = logits_all > 0.5
        ap = average_precision(logits_all, labels_all).cpu().item()
        auc = auroc(logits_all, labels_all).cpu().item()
        acc = accuracy(pred_all, labels_all).cpu().item()

        fprs, tprs, thresholds = roc(logits_all, labels_all)
        fpr_l, tpr_l, thres_l = get_TPR_FPR_metrics(fprs, tprs, thresholds)
        print_tp_fp_thres(args.tasks, logger, fpr_l, tpr_l, thres_l)
        
    print('总推理时间', np.sum(m_infer_time))
    logger.info(attn_weight_all.mean(0))
    encoder.train()
    decoder.train()
    return ap, auc, acc, np.mean(m_loss)

def get_TPR_FPR_metrics(fprs, tprs, thresholds):
    FPR_limits=torch.tensor([0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01])
    fpr_l, tpr_l, thres_l=[], [], []
    for limits in FPR_limits:
        idx = torch.where(fprs>limits)[0][0].item()
        fpr_l.append(fprs[idx])
        tpr_l.append(tprs[idx])
        thres_l.append(thresholds[idx])
    return fpr_l, tpr_l, thres_l

def print_tp_fp_thres(task, logger, fpr_l, tpr_l, thres_l):
    for i in range(len(fpr_l)):
        logger.info('Task {}:  -- FPR: {:.4f}, TPR: {:.4f}, Threshold: {:.4f}'.format(task, fpr_l[i].cpu().item(), tpr_l[i].cpu().item(), thres_l[i].cpu().item()))
    logger.info('---------------------------------------------------------------------')