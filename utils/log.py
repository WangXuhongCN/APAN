import csv
import os
from pathlib import Path
import logging
import time

def result2csv(result, epoch, args, time, mode='test'):
    
    Path("results/").mkdir(parents=True, exist_ok=True)
    if mode == 'val':
        results_path = "results/Test-{}.csv".format(args.prefix)
    elif mode == 'test':
        results_path = "results/Final-{}.csv".format(args.prefix)
    result = [*map(lambda x:round(x,4), result)]
    udf = ''
    if args.balance:
        udf += '+balance'
    if args.pretrain:
        udf += '+pretrain'
    if args.warmup:
        udf += '+warmup'
    if args.uniform:
        udf += '+uniform'
    row = [args.data, time, epoch, args.tasks, args.feat_dim, args.bs, args.lr, args.n_layer, args.n_degree, args.dropout, args.edgedrop, args.eventdrop, args.n_mail]+result+[udf]

    if not os.path.exists(results_path):
        with open(results_path,'w') as f:
            result_writer = csv.writer(f)
            result_writer.writerow(['dataset', 'time','best-epoch', 'tasks', 'emb_dim', 'bs','lr', 'n_layer', 'n_degree', 'dropout', 'edgedrop', 'eventdrop', 'n_mail',  \
                    'ap', 'auc', 'acc', 'loss', 'udf'])

    with open(results_path, 'a') as f:
        result_writer = csv.writer(f)
        result_writer.writerow(row)

def set_logger():
    task_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('log/{}.log'.format(task_time))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger