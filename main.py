import sys
sys.path.insert(0, "../")
import json
import GCN_ETHZ.dataset as D
import argparse
import GCN_ETHZ.utils as utils
from pprint import pprint
import torch
import pickle
from GCN_ETHZ.ed_ranker import EDRanker
import csv
import time

import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

# general args
parser.add_argument('--device', type=int,
                    help="GPU device number",
                    default=0)
parser.add_argument("--mode", type=str,
                    help="train or eval",
                    default='train')
parser.add_argument("--model_path", type=str,
                    help="model path to save/load",
                    default='Model/')
parser.add_argument("--output_path", type=str,
                    help="output path to save/load",
                    default='Output1/')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument("--dropout_rate", type=float,
                    help="dropout rate for ranker model",
                    default=0.2)
parser.add_argument("--dropout_GCN", type=float,
                    help="dropout rate for GCN model",
                    default=0.2)
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--n_sample', type=int, default=5, metavar='G',
                    help='how many nega-sample will be picked for every mention')
parser.add_argument("--use_local_epoches", type=int,
                    help="when training within this epoches, use local model only; after that, use global and local model will be used to init eval entity. 0 means global always, -1 means local always",
                    default=400)

# args for preranking (i.e. 2-step candidate selection)
# for every mention, pick n_cands candidate entities before ranking
parser.add_argument("--n_cands_before_rank", type=int,
                    help="number of candidates",
                    default=50)
parser.add_argument("--prerank_ctx_window", type=int,
                    help="size of context window for the preranking model",
                    default=50)
parser.add_argument("--keep_p_e_m", type=int,
                    help="number of top candidates to keep w.r.t p(e|m)",
                    default=4)
parser.add_argument("--keep_ctx_ent", type=int,
                    help="number of top candidates to keep w.r.t using context",
                    default=4)

# args for local model
parser.add_argument("--ctx_window", type=int,
                    help="size of context window for the local model",
                    default=100)

# only choose n contextual words for local-similarity calculating
parser.add_argument("--tok_top_n", type=int,
                    help="number of top contextual words for the local model",
                    default=40)

# args for global model
parser.add_argument("--hid_dims", type=int,
                    help="number of hidden neurons",
                    default=100)
parser.add_argument("--edge_window", type=int,
                    help="if distance between two words is smaller than this value, they will be connected in graph",
                    default=40)
parser.add_argument("--batch_maxsize", type=int,
                    help="max num of mentions in one batch",
                    default=80)
parser.add_argument("--max_nm", type=int,
                    help="max number of n_node*n_ment",
                    default=12000)
parser.add_argument("--change_rate", type=float,
                    help="change how much of ments one time when predicting",
                    default=0.5)

# args for training
parser.add_argument("--n_epochs", type=int,
                    help="max number of training epochs",
                    default=300)
parser.add_argument("--dev_f1_change_lr", type=float,
                    help="dev f1 to change learning rate",
                    default=0.928)
parser.add_argument("--dev_f1_start_order_learning", type=float,
                    help="dev f1 to start order learning",
                    default=0.92)
parser.add_argument("--eval_after_n_epochs", type=int,
                    help="number of epochs to eval",
                    default=5)
parser.add_argument("--learning_rate", type=float,
                    help="learning rate",
                    default=2e-4)
parser.add_argument("--margin", type=float,
                    help="margin",
                    default=0.01)
parser.add_argument("--margin_global", type=float,
                    help="margin_global",
                    default=0.01)
parser.add_argument("--global_beta", type=float,
                    help="the ratio between global score and local score",
                    default=0.5)

parser.add_argument('--search_ment_size', type=int, default=20,
                    help='change entities of how many mention when predict-searching')
parser.add_argument('--search_entity_size', type=int, default=15,
                    help='sample how many cands for every mention when predict-searching')
parser.add_argument('--predict_epoches', type=int, default=20,
                    help='max predict-searching steps')
parser.add_argument('--death_epoches', type=int, default=3,
                    help='stop predict-searching after how many epoches without improvement')
parser.add_argument("--local_stop_epoches", type=int,
                    help="stop local training after how many epoches without improvement",
                    default=2)

parser.add_argument('--one_entity_once', type=int, default=0,
                    help='')

parser.add_argument("--use_early_stop", type=str2bool, nargs='?', default='n', const=True,
                    help="")

args = parser.parse_args()

# if gpu is to be used
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(args.device)

torch.manual_seed(args.seed)    # set random seed for cpu
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)   # set random seed for present GPU

datadir = '../data/generated/test_train_data'
conll_path = '../data/basic_data/test_datasets'
person_path = '../data/basic_data/p_e_m_data/persons.txt'
voca_emb_dir = "../data/generated/embeddings/word_ent_embs/"
ent_inlinks_path = "../data/entityid_dictid_inlinks_uniq.pkl"

timestr = time.strftime("%Y%m%d-%H%M%S")

F1_CSV_Path = args.output_path + "_" + timestr + "_" + "f1.csv"

if __name__ == "__main__":
    print('load conll at', datadir)
    conll = D.CoNLLDataset(datadir, conll_path, person_path, edge_window=args.edge_window, batch_maxsize=args.batch_maxsize)

    print('create model')
    word_voca, word_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.word',
                                                      voca_emb_dir + 'word_embeddings.npy')

    entity_voca, entity_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.entity',
                                                          voca_emb_dir + 'entity_embeddings.npy')

    with open(ent_inlinks_path, 'rb') as f_pkl:
        ent_inlinks_dict = pickle.load(f_pkl)
        
    ent_inlinks_dict = utils.bi_inlinks(ent_inlinks_dict)

    config = {'hid_dims': args.hid_dims,
              'emb_dims': entity_embeddings.shape[1],
              'freeze_embs': True,
#               'freeze_embs': False,
              'tok_top_n': args.tok_top_n,
              'margin': args.margin,
              'margin_global': args.margin_global,
              'word_voca': word_voca,
              'entity_voca': entity_voca,
              'word_embeddings': word_embeddings,
              'entity_embeddings': entity_embeddings,
              'entity_inlinks': ent_inlinks_dict,
              'dr': args.dropout_rate,
              'gdr': args.dropout_GCN,
              'gamma': args.gamma,
              'f1_csv_path': F1_CSV_Path,
              'one_entity_once': args.one_entity_once,
              'n_sample': args.n_sample,
              'search_ment_size': args.search_ment_size,
              'search_entity_size': args.search_entity_size,
              'predict_epoches': args.predict_epoches,
              'death_epoches': args.death_epoches,
              'lr': args.learning_rate, 
              'n_epochs': args.n_epochs, 
              'use_early_stop' : args.use_early_stop,
              'use_local_epoches': args.use_local_epoches,
              'local_stop_epoches': args.local_stop_epoches,
              'global_beta': args.global_beta,
              'change_rate': args.change_rate,
              'model_path': args.model_path,
              'args': args}

    # pprint(config)
    ranker = EDRanker(config=config)

    dev_datasets = [
                    # ('aida-train', conll.train),
                    ('aida-A', conll.testA, conll.testA_mlist, conll.testA_madj),
                    ('aida-B', conll.testB, conll.testB_mlist, conll.testB_madj),
                    ('msnbc', conll.msnbc, conll.msnbc_mlist, conll.msnbc_madj),
                    ('aquaint', conll.aquaint, conll.aquaint_mlist, conll.aquaint_madj),
                    ('ace2004', conll.ace2004, conll.ace2004_mlist, conll.ace2004_madj),
                    ('clueweb', conll.clueweb, conll.clueweb_mlist, conll.clueweb_madj),
                    ('wikipedia', conll.wikipedia, conll.wikipedia_mlist, conll.wikipedia_madj) 
                ]

    with open(F1_CSV_Path, 'w') as f_csv_f1:
        f1_csv_writer = csv.writer(f_csv_f1)
        f1_csv_writer.writerow(['dataset', 'epoch', 'dynamic', 'F1 Score'])

    if args.mode == 'train' or args.mode == 'load_train':
        print('training...')
        ranker.train((conll.train, conll.train_mlist, conll.train_madj), dev_datasets, config)

    elif args.mode == 'eval':
        org_dev_datasets = dev_datasets  # + [('aida-train', conll.train)]
        dev_datasets = []
        for dname, data in org_dev_datasets:
            dev_datasets.append((dname, ranker.get_data_items(data, predict=True)))
            print(dname, '#dev docs', len(dev_datasets[-1][1]))

        for di, (dname, data, mlist, madj) in enumerate(dev_datasets):
            predictions = ranker.predict(data, mlist, madj, isLocal=False)
            f1, precision = D.eval(org_dev_datasets[di][1], predictions)
            print(dname, 'micro F1: ' + str(f1), ", precision: " + str(precision))

