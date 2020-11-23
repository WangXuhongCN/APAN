import torch
import dgl
from dataloader import MultiLayerTemporalNeighborSampler, TemporalEdgeCollator

def dataloader(args, g):
    origin_num_edges = g.num_edges()//2
    # if args.tasks != 'LP': 
    #     # I know it is strange, but TGAT model does not use val dataset in node classification task.
    #     # https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/blob/master/learn_node.py
    #     train_eid = torch.arange(0, int(0.85 * origin_num_edges))
    #     test_eid = val_eid = torch.arange(int(0.85 * origin_num_edges), origin_num_edges)
    # else:
    train_eid = torch.arange(0, int(0.7 * origin_num_edges))
    val_eid = torch.arange(int(0.7 * origin_num_edges), int(0.85 * origin_num_edges))
    test_eid = torch.arange(int(0.85 * origin_num_edges), origin_num_edges)

    # reverse_eids = torch.cat([torch.arange(origin_num_edges, 2 * origin_num_edges), torch.arange(0, origin_num_edges)])
    exclude, reverse_eids = None, None

    negative_sampler = dgl.dataloading.negative_sampler.Uniform(1) if 'LP' in args.tasks else None

    sampler = MultiLayerTemporalNeighborSampler(args, [args.n_degree, args.n_degree], return_eids=False)
    train_collator = TemporalEdgeCollator(g, train_eid, sampler, exclude=exclude, reverse_eids=reverse_eids, negative_sampler=negative_sampler, eventdrop=args.eventdrop)
    
    train_loader = torch.utils.data.DataLoader(
                        train_collator.dataset, collate_fn=train_collator.collate,
                        batch_size=args.bs, shuffle=False, drop_last=False, num_workers=args.n_worker)
    val_collator = TemporalEdgeCollator(g, val_eid, sampler, exclude=exclude, reverse_eids=reverse_eids, negative_sampler=negative_sampler)
    val_loader = torch.utils.data.DataLoader(
                        val_collator.dataset, collate_fn=val_collator.collate,
                        batch_size=args.bs, shuffle=False, drop_last=False, num_workers=args.n_worker)
    test_collator = TemporalEdgeCollator(g, test_eid, sampler, exclude=exclude, reverse_eids=reverse_eids, negative_sampler=negative_sampler)
    test_loader = torch.utils.data.DataLoader(
                        test_collator.dataset, collate_fn=test_collator.collate,
                        batch_size=args.bs, shuffle=False, drop_last=False, num_workers=args.n_worker)
    return train_loader, val_loader, test_loader, val_eid.shape[0], test_eid.shape[0]