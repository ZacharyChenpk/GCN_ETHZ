import numpy as np
from GCN_ETHZ.vocabulary import Vocabulary
import torch
from torch.autograd import Variable
import GCN_ETHZ.dataset as D
import GCN_ETHZ.utils as utils
import GCN_ETHZ.ntee as ntee
from GCN_ETHZ.gcn.model import GCN
from random import shuffle
import torch.optim as optim
from GCN_ETHZ.abstract_word_entity import load as load_model
from GCN_ETHZ.mulrel_ranker import MulRelRanker
from pprint import pprint
from itertools import count
import copy
import csv
import json
import time
from collections import Counter
# import ipdb
# import pdb
import random
import math

ModelClass = MulRelRanker
wiki_prefix = 'en.wikipedia.org/wiki/'
debugging = False
debugging_skip_train = False

class EDRanker:
    """
    ranking candidates
    """
    def __init__(self, config):
        print('--- create EDRanker model ---')

        config['entity_embeddings'] = config['entity_embeddings'] / \
                                      np.maximum(np.linalg.norm(config['entity_embeddings'],
                                                                axis=1, keepdims=True), 1e-12)
        config['entity_embeddings'][config['entity_voca'].unk_id] = 1e-10
        config['word_embeddings'] = config['word_embeddings'] / \
                                    np.maximum(np.linalg.norm(config['word_embeddings'],
                                                              axis=1, keepdims=True), 1e-12)
        config['word_embeddings'][config['word_voca'].unk_id] = 1e-10
        self.one_entity_once = config['one_entity_once']
        self.ent_inlinks = config['entity_inlinks']
        self.n_sample = config['n_sample']
        self.search_ment_size = config['search_ment_size']
        self.search_entity_size = config['search_entity_size']
        self.predict_epoches = config['predict_epoches']
        self.death_epoches = config['death_epoches']
        self.local_stop_epoches = config['local_stop_epoches']
        self.change_rate = config['change_rate']
        self.local_cur_stop_epoches = 0

        self.word_vocab = config['word_voca']
        self.ent_vocab = config['entity_voca']

        self.output_path = config['f1_csv_path']
        print('prerank model')
        self.prerank_model = ntee.NTEE(config)
        self.args = config['args']

        print('main model')
        if self.args.mode == 'eval' or self.args.mode == 'load_train':
            print('try loading model from', self.args.model_path)
            self.model = load_model(self.args.model_path, ModelClass)
        else:
            print('create new model')

            config['use_local_epoches'] = self.args.use_local_epoches
            config['oracle'] = False
            self.model = ModelClass(config)

        self.load_ent_desc(500, 3)

        self.prerank_model.cuda()
        self.emb_dims = self.model.emb_dims
        self.negsam_graph_cache = {}
        self.true_cands = {}
        # self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()

    def load_ent_desc(self, max_desc_len, n_grams):
        ent_desc = json.load(open('../data/ent2desc.json', 'r'))
        self.ent_desc = [[self.word_vocab.get_id(Vocabulary.unk_token) for j in range(max_desc_len)] for i in range(self.ent_vocab.size())]
        self.desc_mask = [[0 for j in range(max_desc_len-n_grams+1)] for i in range(self.ent_vocab.size())]
        for ent in ent_desc:
            for i in range(min(len(ent_desc[ent]), max_desc_len)):
                self.ent_desc[self.ent_vocab.get_id(ent)][i] = self.word_vocab.get_id(ent_desc[ent][i])
                if (i>=n_grams-1):
                    # seems that only pick the last n-gram words?
                    self.desc_mask[self.ent_vocab.get_id(ent)][i-(n_grams-1)] = 1
        self.ent_desc = Variable(torch.LongTensor(self.ent_desc).cuda())
        self.desc_mask = Variable(torch.LongTensor(self.desc_mask).cuda())


    def get_data_items(self, dataset, predict=False, isTrain=False):
        data = []
        cand_source = 'candidates'

        for doc_name, content in dataset.items():
            items = []

            for m in content:
                try:
                    named_cands = [c[0] for c in m[cand_source]]
                    p_e_m = [min(1., max(1e-3, c[1])) for c in m[cand_source]]
                    etype = [c[2] for c in m[cand_source]]
                except:
                    named_cands = [c[0] for c in m['candidates']]
                    p_e_m = [min(1., max(1e-3, c[1])) for c in m['candidates']]
                    etype = [c[2] for c in m['candidates']]
                try:
                    # true_pos is the index of true answer in the list of candidates (named_cands)
                    true_pos = named_cands.index(m['gold'][0])
                    p = p_e_m[true_pos]
                except:
                    true_pos = -1

                # select # n_cands_before_rank candidates according to their priority
                named_cands = named_cands[:min(self.args.n_cands_before_rank, len(named_cands))]
                p_e_m = p_e_m[:min(self.args.n_cands_before_rank, len(p_e_m))]
                etype = etype[:min(self.args.n_cands_before_rank, len(etype))]
                # guarantee that the ground truth is in the top30 candidates
                ### why they put the last top-cand as the fake truePos?
                if true_pos >= len(named_cands):
                    if not predict:
                        true_pos = len(named_cands) - 1
                        p_e_m[-1] = p
                        named_cands[-1] = m['gold'][0]
                    else:
                        true_pos = -1
                        
                if true_pos < 0 and not predict:
                    true_pos = 0
                
                # cands is the dict_index of candidates
                cands = [self.model.entity_voca.get_id(wiki_prefix + c) for c in named_cands]
                mask = [1.] * len(cands)

                if len(cands) == 0 and not predict:
                    continue
                # if cands is not enough, pad it
                elif len(cands) < self.args.n_cands_before_rank:
                    cands += [self.model.entity_voca.unk_id] * (self.args.n_cands_before_rank - len(cands))
                    etype += [[0, 0, 0, 1]] * (self.args.n_cands_before_rank - len(etype))
                    named_cands += [Vocabulary.unk_token] * (self.args.n_cands_before_rank - len(named_cands))
                    p_e_m += [1e-8] * (self.args.n_cands_before_rank - len(p_e_m))
                    mask += [0.] * (self.args.n_cands_before_rank - len(mask))

                
                lctx = m['context'][0].strip().split()
                lctx_ids = [self.prerank_model.word_voca.get_id(t) for t in lctx if utils.is_important_word(t)]
                lctx_ids = [tid for tid in lctx_ids if tid != self.prerank_model.word_voca.unk_id]
                lctx_ids = lctx_ids[max(0, len(lctx_ids) - self.args.ctx_window//2):]

                rctx = m['context'][1].strip().split()
                rctx_ids = [self.prerank_model.word_voca.get_id(t) for t in rctx if utils.is_important_word(t)]
                rctx_ids = [tid for tid in rctx_ids if tid != self.prerank_model.word_voca.unk_id]
                rctx_ids = rctx_ids[:min(len(rctx_ids), self.args.ctx_window//2)]

                # seem to be one word at most time?
                # it's 'mention' with conll_m but not 'mentions' in conll_doc
                ment = m['mention'].strip().split()
                ment_ids = [self.prerank_model.word_voca.get_id(t) for t in ment if utils.is_important_word(t)]
                ment_ids = [tid for tid in ment_ids if tid != self.prerank_model.word_voca.unk_id]

                m['sent'] = ' '.join(lctx + rctx)
                mtype = m['mtype']
                items.append({'context': (lctx_ids, rctx_ids),
                              'ment_ids': ment_ids,
                              'cands': cands,
                              'named_cands': named_cands,
                              'p_e_m': p_e_m,
                              'mask': mask,
                              'true_pos': true_pos,
                              'mtype': mtype,
                              'etype': etype,
                              'doc_name': doc_name,
                              'raw': m,
                              'prev_dist': m['prev_dist']
                              })

            # one doc one time here
            if len(items) > 0:
                # note: this shouldn't affect the order of prediction because we use doc_name to add predicted entities,
                # and we don't shuffle the data for prediction
                
                # # ----old implementation----
                # # each doc is regarded as one batch
                # # data.append(items)
                #     if isTrain:
                #         for k in range(0, len(items), self.seq_len // 2):
                #             data.append(items[max(0, k - self.seq_len//2) : min(len(items), k + self.seq_len//2)])
                #     else:
                #         if self.one_entity_once:
                #             for k in range(0, len(items)):
                #                 data.append(items[max(0, k-self.seq_len+1): k+1])
                #         else:
                #             for k in range(0, len(items), self.seq_len):
                #                 data.append(items[k:min(len(items), k + self.seq_len)])

                # in order to use GCN, we try to process one whole doc for one time 
                ### New ###
                data.append(items)
                ###     ###

        # every element in 'data' is a list of items
        return self.prerank(data, predict)
    
    # executed between the get_data_item and training
    def prerank(self, dataset, predict=False):
        new_dataset = []
        has_gold = 0
        total = 0

        for content in dataset:
            # content is a list of items
            items = []

            if self.args.keep_ctx_ent > 0:
                # rank the candidates by ntee scores
                lctx_ids = [m['context'][0][max(len(m['context'][0]) - self.args.prerank_ctx_window // 2, 0):]
                            for m in content]
                rctx_ids = [m['context'][1][:min(len(m['context'][1]), self.args.prerank_ctx_window // 2)]
                            for m in content]
                ment_ids = [[] for m in content]

                token_ids = [l + m + r if len(l) + len(r) > 0 else [self.prerank_model.word_voca.unk_id]
                             for l, m, r in zip(lctx_ids, ment_ids, rctx_ids)]

                entity_ids = [m['cands'] for m in content]
                entity_ids = Variable(torch.LongTensor(entity_ids).cuda())

                entity_mask = [m['mask'] for m in content]
                entity_mask = Variable(torch.FloatTensor(entity_mask).cuda())

                # token_ids' element is a part of the document which around the target mention
                token_ids, token_offsets = utils.flatten_list_of_lists(token_ids)
                token_ids = Variable(torch.LongTensor(token_ids).cuda())
                token_offsets = Variable(torch.LongTensor(token_offsets).cuda())

                # log_probs seem to be n_ments * n_cands ?
                log_probs = self.prerank_model.forward(token_ids, token_offsets, entity_ids, use_sum=True)
                log_probs = (log_probs * entity_mask).add_((entity_mask - 1).mul_(1e10))
                # top_pos seem to be n_ments * k ?
                _, top_pos = torch.topk(log_probs, dim=1, k=self.args.keep_ctx_ent)
                top_pos = top_pos.data.cpu().numpy()
            else:
                top_pos = [[]] * len(content)

            # select candidats: mix between keep_ctx_ent best candidates (ntee scores) with
            # keep_p_e_m best candidates (p_e_m scores)
            for i, m in enumerate(content):
                sm = {'cands': [],
                      'named_cands': [],
                      'p_e_m': [],
                      'mask': [],
                      'etype': [],
                      'true_pos': -1}
                m['selected_cands'] = sm

                selected = set(top_pos[i])
                idx = 0
                # insert remained candidate until fit the requirement
                while len(selected) < self.args.keep_ctx_ent + self.args.keep_p_e_m:
                    if idx not in selected:
                        selected.add(idx)
                    idx += 1

                selected = sorted(list(selected))
                for idx in selected:
                    if idx>len(m['cands'])-1:
                        continue
                    sm['cands'].append(m['cands'][idx])
                    sm['named_cands'].append(m['named_cands'][idx])
                    sm['p_e_m'].append(m['p_e_m'][idx])
                    sm['mask'].append(m['mask'][idx])
                    sm['etype'].append(m['etype'][idx])
                    # record the answer's position in sm
                    if idx == m['true_pos']:
                        sm['true_pos'] = len(sm['cands']) - 1

                if not predict:
                    if sm['true_pos'] == -1:
                        continue

                items.append(m)
                if sm['true_pos'] >= 0:
                    has_gold += 1
                total += 1

                if predict:
                    # only for oracle model, not used for eval
                    # h? why 'predict' means oracle model but not eval?
                    if sm['true_pos'] == -1:
                        sm['true_pos'] = 0  # a fake gold, happens only 2%, but avoid the non-gold

            if len(items) > 0:
                new_dataset.append(items)

        print('recall', has_gold / total)
        return new_dataset

    def gold_e_graph_build(self, dataset):
        cand_to_idxs = []
        idx_to_cands = []
        gold_e_adjs = []
        for dc, batch in enumerate(dataset):
            sele_cand = [m['selected_cands']['cands'] for m in batch]
            true_pos = [m['selected_cands']['true_pos'] for m in batch]
            true_cands = [sele_cand[idx][true_pos[idx]] if true_pos[idx] > -1 else sele_cand[idx][0] for idx in range(len(true_pos))]
            self.true_cands[dc] = true_cands
            cand_to_idx, idx_to_cand, e_adj = self.e_graph_build(true_cands)
            cand_to_idxs.append(cand_to_idx)
            idx_to_cands.append(idx_to_cand)
            gold_e_adjs.append(e_adj)
        return cand_to_idxs, idx_to_cands, gold_e_adjs

    def e_graph_build(self, cand_ids, hops=2, max_nm=False):
        if max_nm == False:
            max_nm = self.args.max_nm
        cand_to_idx = {}
        idx_to_cand = copy.deepcopy(cand_ids)
        node_counter = Counter()
        max_cand = max(self.ent_inlinks.keys())
        cnt = 0
        n_ments = len(cand_ids)
        max_n_node = int(max_nm / n_ments)
        if hops == 2:
            for c in cand_ids:
                if c not in cand_to_idx:
                    cand_to_idx[c] = []
                    if c in self.ent_inlinks:
                        neighbor = self.ent_inlinks[c]
                        node_counter.update(neighbor)
                cand_to_idx[c].append(cnt)
                cnt+=1
#                 if c in self.ent_inlinks:
#                     neighbor = self.ent_inlinks[c]
#                     node_counter.update(neighbor)
            for n in list(node_counter.elements()):
                if node_counter[n] > 1 and (n not in cand_to_idx) and n < self.model.entity_voca.size():
                    if n not in cand_to_idx:
                        cand_to_idx[n] = []
                    cand_to_idx[n].append(cnt)
                    idx_to_cand.append(n)
                    cnt+=1
                    if cnt >= max_n_node:
                        break
            n = len(idx_to_cand)
            e_adj = np.zeros((n, n))
            for cand, idx in cand_to_idx.items():
                n_idx = len(idx)
                slic = e_adj[idx]
                slic[:,idx] = np.ones((n_idx,n_idx))-np.eye(n_idx)
                e_adj[idx] = slic
                if cand not in self.ent_inlinks:
                    # print(cand, "not in ent_inlinks")
                    continue
                neighbor = [nid for a in self.ent_inlinks[cand] if a in cand_to_idx for nid in cand_to_idx[a]]
                slic = e_adj[idx]
                slic[:,neighbor]=1
                e_adj[idx] = slic
                slic = e_adj[neighbor]
                slic[:,idx]=1
                e_adj[neighbor] = slic
            return cand_to_idx, idx_to_cand, e_adj
        elif hops == 1:
            for c in cand_ids:
                cand_to_idx[c] = len(cand_to_idx)
            n = len(idx_to_cand)
            e_adj = np.zeros((n, n))
            for cand, idx in cand_to_idx.items():
                if cand not in self.ent_inlinks:
                    continue
                neighbor = [cand_to_idx[a] for a in self.ent_inlinks[cand] if a in cand_to_idx]
                e_adj[idx, neighbor] = 1
                e_adj[neighbor, idx] = 1
            return cand_to_idx, idx_to_cand, e_adj

    def ment_neg_sample(self, n_sample, entity_ids, true_pos, entity_mask):
        # entity_ids is n_ment * n_cand(k) indexes
        n_ment, n_cand = entity_ids.size()
        
        copy_entity_mask = entity_mask.index_put((torch.LongTensor(np.arange(n_ment)).cuda(), true_pos.long()), torch.zeros(n_ment).cuda())
        only_one_cand = (copy_entity_mask.sum(dim=1)==0).float().cuda()
        copy_entity_mask = entity_mask.index_put((torch.LongTensor(np.arange(n_ment)).cuda(), true_pos.long()), only_one_cand)
        sample_idx = torch.multinomial(copy_entity_mask, n_sample, replacement=True)
        
        i_matrix = torch.LongTensor(list(range(n_ment))).cuda().unsqueeze(1).repeat(1, n_sample)
        only_one_cand = only_one_cand.unsqueeze(1).repeat(1, n_sample)
        copy_entity_mask.index_put_((torch.flatten(i_matrix), torch.flatten(sample_idx)), torch.flatten(only_one_cand))
        # sample_idx is n_ment * n_sample
        # copy_entity_mask: n_ment * n_cand
        return sample_idx, copy_entity_mask

#     def ment_neg_sample(self, n_sample, entity_ids, true_pos, entity_mask, change_rate):
#         # entity_ids is n_ment * n_cand(k) indexes
#         n_ment, n_cand = entity_ids.size()
#         ment_res_num = torch.FloatTensor([n_sample] * n_ment).cuda()
#         assert change_rate <= 1.0
#         change_one_time = max(int(change_rate * n_ment), 1)
#         copy_entity_mask = entity_mask.index_put((torch.LongTensor(np.arange(n_ment)).cuda(), true_pos.long()), torch.zeros(n_ment).cuda())
#         only_one_cand = (copy_entity_mask.sum(dim=1)==0).float().cuda()
#         copy_entity_mask = entity_mask.index_put((torch.LongTensor(np.arange(n_ment)).cuda(), true_pos.long()), only_one_cand)

#         neg_sample_idx = true_pos.unsqueeze(0).unsqueeze(0).repeat(n_ment, n_sample, 1)
#         copy_entity_ids = entity_ids.t().squeeze(0).repeat(n_ment, 1, 1)
#         # neg_sample_idx: n_ment * n_sample * n_ment
#         # copy_entity_ids: n_ment * n_cand * n_ment

#         while ment_res_num.sum() > 0:
#             sample_ment_idx = torch.multinomial(ment_res_num, min(change_one_time, (ment_res_num>0).sum()), replacement=False)
#             sample_mask = copy_entity_mask[sample_ment_idx]
#             sample_ent_idx = torch.multinomial(sample_mask, 1, replacement=False).squeeze(1)
# #             print(sample_ment_idx, sample_ent_idx)
#             neg_sample_idx[:,:,sample_ment_idx] = neg_sample_idx[:,:,sample_ment_idx].index_put((sample_ment_idx, n_sample-ment_res_num[sample_ment_idx].long()), sample_ent_idx)
#             #print(neg_sample_idx[:,:,sample_ment_idx])
#             ment_res_num[sample_ment_idx] -= 1

#         return neg_sample_idx, torch.gather(copy_entity_ids, 1, neg_sample_idx)

    # Heuristic Order Learning Method - Based on Mention-Local Similarity or Mention-Topical Similarity
    def train(self, org_train_dataset, org_dev_datasets, config):
        print('extracting training data')
        org_train_dataset, train_mlist, train_madj = org_train_dataset
        train_dataset = self.get_data_items(org_train_dataset, predict=False, isTrain=True)

        doc_names = list(train_mlist.keys())
        shuffle_list = list(zip(train_dataset, doc_names))
        shuffle(shuffle_list)
        train_dataset[:], doc_names[:] = zip(*shuffle_list)
        train_dataset_adj = utils.data_m_graph_build(train_dataset)
        
        gold_cand_to_idxs, gold_idx_to_cand, gold_e_adjs = self.gold_e_graph_build(train_dataset)
        print('#train docs', len(train_dataset))
        self.init_lr = config['lr']
        dev_datasets = []
        for dname, data, mlist, madj in org_dev_datasets:
            dataitems = self.get_data_items(data, predict=True, isTrain=False)
            dev_datasets.append((dname, dataitems, mlist, utils.data_m_graph_build(dataitems)))
            print(dname, '#dev docs', len(dev_datasets[-1][1]))

        print('creating optimizer')
        optimizer_local = optim.Adam(self.model.local_parameter(), lr=config['lr'])
        optimizer_global = optim.Adam(self.model.global_parameter(), lr=config['lr'], weight_decay=0.02)

        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                print(param_name)

        best_f1 = -1
        not_better_count = 0
        is_counting = True
        eval_after_n_epochs = self.args.eval_after_n_epochs

        order_learning = False
        # order_learning_count = 0

        rl_acc_threshold = 0.7

        # optimize the parameters within the disambiguation module first
        # self.model.switch_order_learning(0)
        best_aida_A_rlts = []
        best_aida_A_f1 = 0.
        best_aida_B_rlts = []
        best_aida_B_f1 = 0.
        best_ave_rlts = []
        best_ave_f1 = 0.
        
        first_global_flag = True

        self.run_time = []
        for e in range(config['n_epochs']):
            
            if debugging_skip_train:
                for di, (dname, data, mlist, madj) in enumerate(dev_datasets):
                    if dname == 'aida-B':
                        self.rt_flag = True
                    else:
                        self.rt_flag = False
                    # pdb.set_trace()
                    predictions = self.predict(data, mlist, madj, isLocal=False)
                    #self.records[e][dname] = self.record
                    f1, precisioin = D.eval(org_dev_datasets[di][1], predictions)

                    print(dname, 'micro F1:', str(f1), ", precisioin:", str(precisioin), flush=True)
                break

            total_loss = 0
            cur_max_n_ment = 0
            start_time = time.time()
            isLocal = self.local_only_judge(e)

            for dc, batch in enumerate(train_dataset):  # each document is a minibatch
                self.model.train()
                # print("dc:",dc,"start")
                if first_global_flag:
                    dc_start_time = time.time()
                
                # convert data items to pytorch inputs
                token_ids = [m['context'][0] + m['context'][1]
                             if len(m['context'][0]) + len(m['context'][1]) > 0
                             else [self.model.word_voca.unk_id]
                             for m in batch]

                ment_ids = [m['ment_ids'] if len(m['ment_ids']) > 0
                            else [self.model.word_voca.unk_id]
                            for m in batch]

                entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).cuda())
                true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).cuda())
                p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).cuda())
                entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).cuda())

                # entity_ids is n_ment * n_cand(k) indexes
                # for every entity candidate, we have a description
                desc_ids = torch.index_select(self.ent_desc, 0, entity_ids.view(-1)).view(entity_ids.size(0), entity_ids.size(1), -1)
                desc_mask = torch.index_select(self.desc_mask, 0, entity_ids.view(-1)).view(entity_ids.size(0), entity_ids.size(1), -1)

                mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).cuda())
                etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).cuda())

                token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)
                token_ids = Variable(torch.LongTensor(token_ids).cuda())
                token_mask = Variable(torch.FloatTensor(token_mask).cuda())

                ment_ids, ment_mask = utils.make_equal_len(ment_ids, self.model.word_voca.unk_id)
                ment_ids = Variable(torch.LongTensor(ment_ids).cuda())
                ment_mask = Variable(torch.FloatTensor(ment_mask).cuda())
                
                optimizer_local.zero_grad()
                optimizer_global.zero_grad()
                # optimizer.zero_grad()

                n_ment, n_cand = entity_ids.size()
                cur_max_n_ment = max(cur_max_n_ment, n_ment)
                # print("cur_max_n_ment:", cur_max_n_ment)
                e_cand_to_idxs = [[] for _ in range(n_ment) ]
                e_idx_to_cands = [[] for _ in range(n_ment) ]
                e_adjs = [[] for _ in range(n_ment) ]
                sele_cand = [m['selected_cands']['cands'] for m in batch]
                
#                 # sample_idx: n_ment * n_sample
#                 # copy_entity_mask: n_ment * n_cand
                sample_idx, copy_entity_mask = self.ment_neg_sample(self.n_sample, entity_ids, true_pos, entity_mask)

                # neg_sample_idx: n_ment * n_sample * n_ment
                # neg_sample_cands: n_ment * n_sample * n_ment
                # sample_idx: n_ment * n_sample
#                 neg_sample_idx, neg_sample_cands = self.ment_neg_sample(self.n_sample, entity_ids, true_pos, entity_mask, self.change_rate)
#                 sample_idx = torch.gather(neg_sample_idx, 2, torch.LongTensor(range(n_ment)).cuda().unsqueeze(1).unsqueeze(2).repeat(1, self.n_sample, 1)).squeeze(2)

                if isLocal == False:
                    ###  USE LOCAL AND GLOBAL
                    for i in range(n_ment):
                        for j in range(self.n_sample):
                            true_cands = copy.deepcopy(self.true_cands[dc])
                            true_cands[i] = sele_cand[i][sample_idx[i][j]]
                            cand_to_idx, idx_to_cand, e_adj = self.e_graph_build(true_cands)
#                             cand_to_idx, idx_to_cand, e_adj = self.e_graph_build(neg_sample_cands[i][j].tolist())
                            e_cand_to_idxs[i].append(cand_to_idx)
                            e_idx_to_cands[i].append(idx_to_cand)
                            e_adjs[i].append(e_adj)
                        e_cand_to_idxs[i].append(gold_cand_to_idxs[dc])
                        e_idx_to_cands[i].append(gold_idx_to_cand[dc])
                        e_adjs[i].append(gold_e_adjs[dc])

                    # e_adjs = torch.LongTensor(e_adjs)
                    # e_adjs: n_ment * n_sample * n_entity * n_entity
                    new_adjs, new_node_cands, new_node_mask = utils.e_graph_batch_padding(e_cand_to_idxs, e_idx_to_cands, e_adjs, n_ment, self.n_sample)
                    # new_adjs: n_ment * (n_sample+1) * n_node * n_node
                    # new_node_cands: n_ment * (n_sample+1) * n_node
                    # new_node_mask: n_ment * (n_sample+1) * n_node
                    nega_e = (new_adjs, new_node_cands, new_node_mask)

                    cur_doc_name = doc_names[dc]
                    
                    if first_global_flag:
                        dc_end_time = time.time()
                        dc_n_ment = new_node_mask.size(0)
                        dc_n_node = new_node_mask.size(2)
                        print("dc", dc, "n_node:", dc_n_node, ",n_ment:", dc_n_ment, ",n1m:", dc_n_node*dc_n_ment, ",n2m:", dc_n_node*dc_n_node*dc_n_ment, ",preparing use time:", dc_end_time-dc_start_time, flush=True)

                    scores, dual_scores = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype, ment_ids, ment_mask, desc_ids, desc_mask, train_mlist[cur_doc_name], train_dataset_adj[dc], nega_e, sample_idx, gold=true_pos.view(-1, 1), isTrain=True, isLocal=False)
#                     if first_global_flag:
#                         dc_end_time = time.time()
#                         print("dc", dc,"total use time:", dc_end_time-dc_start_time, ",avg time:", (dc_end_time-start_time)/(dc+1), flush=True)
                    if dc == 0:
                        # print("global+local:",scores)
                        end_time = time.time()
                        print("dc0 local:",dual_scores[0])
                        print("global:",dual_scores[1])

                    # loss = self.model.loss(dual_scores[1], torch.LongTensor([self.n_sample]*n_ment).cuda(), isLocal=isLocal)
                    loss = self.model.loss(scores, torch.LongTensor([self.n_sample]*n_ment).cuda(), isLocal=isLocal)
                    loss.backward()
#                     if config['model_path'] != 'model_param9':
#                         optimizer_local.step()
                    optimizer_global.step()

                else:
                    ### USE LOCAL ONLY
                    scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype, ment_ids, ment_mask, desc_ids, desc_mask, False, False, False, sample_idx, gold=true_pos.view(-1, 1), isTrain=True, isLocal=True)

                    if self.local_cur_stop_epoches == self.local_stop_epoches - 1 and dc == 0 and (e + 1) % eval_after_n_epochs == 0:
                        print("final local scores:", scores)
                    
                    loss = self.model.loss(scores, true_pos, isLocal=isLocal)
                    loss.backward()
                    optimizer_local.step()

                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=40, norm_type=2)
#                 for (name, p) in self.model.named_parameters():
#                     if name == "att_mat_diag":
#                         # print(name, p.grad.data)
#                         if torch.isnan(p.grad.data).any():
#                             print("att_mat_diag NAN detected")
#                             ipdb.set_trace()
                            # p.grad.data.zero_()
#                             elif torch.sum(p.grad.data)==0:
#                                 print("sum 0 detected")
#                                 ipdb.set_trace()
                
                self.model.regularize(max_norm=2)

                loss = loss.cpu().data.numpy()
                total_loss += loss

                torch.cuda.empty_cache()

            end_time = time.time()
            print('epoch', e, 'total loss', total_loss, total_loss / len(train_dataset), "use time:", end_time-start_time, flush=True)
            if isLocal == False:
                first_global_flag = False

            if (e + 1) % eval_after_n_epochs == 0 or debugging:
                start_time = time.time()
                dev_f1 = 0.
                test_f1 = 0.
                ave_f1 = 0.
                if rl_acc_threshold < 0.92:
                    rl_acc_threshold += 0.02
                temp_rlt = []
                #self.records[e] = dict()
                for di, (dname, data, mlist, madj) in enumerate(dev_datasets):
                    if dname == 'aida-B':
                        self.rt_flag = True
                    else:
                        self.rt_flag = False
                    # predictions = self.predict(data, config['isDynamic'], order_learning)
                    predictions = self.predict(data, mlist, madj, isLocal)
                    #self.records[e][dname] = self.record
                    f1, precision = D.eval(org_dev_datasets[di][1], predictions)

                    if isLocal:
                        print("using local now, waiting for global later")
                    else:
                        print("using global now")
                    print(dname, 'micro F1: ' + str(f1), ', precision: ' + str(precision), flush=True)

                    with open(self.output_path, 'a') as eval_csv_f1:
                        eval_f1_csv_writer = csv.writer(eval_csv_f1)
                        eval_f1_csv_writer.writerow([dname, e, 0, f1])
                    temp_rlt.append([dname, f1])
                    if dname == 'aida-A':
                        dev_f1 = f1
                    if dname == 'aida-B':
                        test_f1 = f1
                    ave_f1 += f1
                if dev_f1>best_aida_A_f1:
                    best_aida_A_f1 = dev_f1
                    best_aida_A_rlts = copy.deepcopy(temp_rlt)
                if test_f1>best_aida_B_f1:
                    best_aida_B_f1 = test_f1
                    best_aida_B_rlts = copy.deepcopy(temp_rlt)
                if ave_f1 > best_ave_f1:
                    best_ave_f1 = ave_f1
                    best_ave_rlts = copy.deepcopy(temp_rlt)

                # if not config['isDynamic']:
                #     self.record_runtime('DCA')
                # else:
                if isLocal:
                    self.record_runtime('local')

                    if dev_f1 < best_f1:
                        self.local_cur_stop_epoches += 1
                    else:
                        self.local_cur_stop_epoches = 0
                        best_f1 = dev_f1
                        print('save model to', self.args.model_path)
                        self.model.save(self.args.model_path)
                else:
                    self.record_runtime('GCN')
                    print('save model to', self.args.model_path+'_'+str(e))
                    self.model.save(self.args.model_path+'_'+str(e))
                    

                self.model.print_weight_norm()
                end_time = time.time()
                print("eval use time:", end_time-start_time)

        print('best_aida_A_rlts', best_aida_A_rlts)
        print('best_aida_B_rlts', best_aida_B_rlts)
        print('best_ave_rlts', best_ave_rlts)

    def record_runtime(self, method):
        self.run_time.sort(key=lambda x:x[0])
        pre_cands = 0
        count = 0
        total = 0.
        rt = dict()
        for cands, ti in self.run_time:
            if not cands==pre_cands:
                if pre_cands > 0:
                    rt[pre_cands] = total / count
                total = ti
                count = 1
                pre_cands = cands
            else:
                count += 1
                total += ti
        if count >0:
            rt[pre_cands] = total / count
        with open('runtime_%s.csv'%method, 'w') as runtime_csv:
            runtime_csv_writer = csv.writer(runtime_csv)
            for cands, ti in rt.items():
                runtime_csv_writer.writerow([cands, ti])
            runtime_csv.close()

    def predict(self, data, mlist, madj, isLocal=False):
        search_ment_size = self.search_ment_size
        search_entity_size = self.search_entity_size
        predict_epoches = self.predict_epoches
        death_epoches = self.death_epoches
        predictions = {items[0]['doc_name']: [] for items in data}
        self.model.eval()
        #self.record = []
        total_changed = 0
        total_ments = 0
        for dc, batch in enumerate(data):  # each document is a minibatch, is a list of mentions
            self.model.doc_predict_restore = False
            start_time = time.time()
            token_ids = [m['context'][0] + m['context'][1]
                         if len(m['context'][0]) + len(m['context'][1]) > 0
                         else [self.model.word_voca.unk_id]
                         for m in batch]

            ment_ids = [m['ment_ids'] if len(m['ment_ids']) > 0
                        else [self.model.word_voca.unk_id]
                        for m in batch]

            total_candidates = sum([len(m['selected_cands']['cands']) for m in batch])

            entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).cuda())
            p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).cuda())
            entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).cuda())
            true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).cuda())

            token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)

            token_ids = Variable(torch.LongTensor(token_ids).cuda())
            token_mask = Variable(torch.FloatTensor(token_mask).cuda())

            desc_ids = torch.index_select(self.ent_desc, 0, entity_ids.view(-1)).view(entity_ids.size(0),entity_ids.size(1), -1)
            desc_mask = torch.index_select(self.desc_mask, 0, entity_ids.view(-1)).view(entity_ids.size(0),entity_ids.size(1), -1)
            ment_ids, ment_mask = utils.make_equal_len(ment_ids, self.model.word_voca.unk_id)
            ment_ids = Variable(torch.LongTensor(ment_ids).cuda())
            ment_mask = Variable(torch.FloatTensor(ment_mask).cuda())

            mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).cuda())
            etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).cuda())

            n_ments, n_cands = entity_ids.size()
            # the val in cur_cand_idxs should be in 0~n_cand
            maybe_no_cand = (entity_mask.sum(dim=1)==0)
            maybe_no_cand_idx = torch.arange(n_ments)[maybe_no_cand]
            if maybe_no_cand.any() and not isLocal:
#                 print("sum of entity_mask 0 detected")
                entity_mask[maybe_no_cand_idx,0] = 1
#                 print("mids:", maybe_no_cand_idx)
#                 print("eids:", entity_ids[maybe_no_cand])
                
                
            scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype, ment_ids, ment_mask, desc_ids, desc_mask, False, False, False, False, gold=False, isTrain=False, isLocal=True)
            final_pde = 0
            if isLocal:
                scores = scores.cpu().data.numpy()

                pred_ids = np.argmax(scores, axis=1)
                
            else:
                cur_cand_idxs = torch.argmax(scores, dim=1)
                local_cur_cands = torch.gather(entity_ids, 1, cur_cand_idxs.unsqueeze(1))

                death_cnt = 0
                print_flag = False # for debug
                for pde in range(predict_epoches):
                    cur_cands = torch.gather(entity_ids, 1, cur_cand_idxs.unsqueeze(1))
                    list_cands = cur_cands.squeeze(1).cpu().numpy().tolist()
                    # cur_cands: n_ment * 1
                    cand_to_idx, idx_to_cand, e_adj = self.e_graph_build(list_cands)
                    idx_to_cand = torch.LongTensor(idx_to_cand).cuda()
                    the_mask = torch.ones(idx_to_cand.size(0)).cuda()
                    nega_e = (torch.LongTensor(e_adj).cuda(), idx_to_cand, the_mask)
                    cur_scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype, ment_ids, ment_mask, desc_ids, desc_mask, None, madj[dc], nega_e, cur_cand_idxs, gold=None, isTrain=False, chosen_ment=False, isLocal=False)
                    assert cur_scores.size(0) == n_ments
                    
                    if print_flag:
                        print_flag = False
                        print("changed new_scores:", cur_scores)

                    # small_scores, small_idxs = torch.topk(cur_scores.squeeze(1), min(search_ment_size, n_ments), largest=False, sorted=True)
                    small_idxs = torch.multinomial(torch.ones(n_ments).cuda(), min(search_ment_size, n_ments))
                    small_scores = cur_scores.squeeze(1)[small_idxs]

                    random_new_cand_idx = torch.cat([cur_cand_idxs[small_idxs].unsqueeze(1), torch.multinomial(entity_mask[small_idxs], min(n_cands, search_entity_size)-1, replacement=True)], dim=1)
                    # small_scores: search_ment_size
                    # random_new_cand_idx: search_ment_size * search_entity_size, value in 0~n_cands
                    e_cand_to_idxs = [[] for _ in range(min(search_ment_size, n_ments)) ]
                    e_idx_to_cands = [[] for _ in range(min(search_ment_size, n_ments)) ]
                    e_adjs = [[] for _ in range(min(search_ment_size, n_ments)) ]

                    for i in range(min(search_ment_size, n_ments)):
                        for j in range(min(n_cands, search_entity_size)):
                            true_cands = copy.deepcopy(list_cands)
                            which_ment = small_idxs[i]
                            true_cands[which_ment] = int(entity_ids[which_ment][random_new_cand_idx[i][j]])
                            cand_to_idx, idx_to_cand, e_adj = self.e_graph_build(true_cands)
                            e_cand_to_idxs[i].append(cand_to_idx)
                            e_idx_to_cands[i].append(idx_to_cand)
                            e_adjs[i].append(e_adj)
                    new_adjs, new_node_cands, new_node_mask = utils.e_graph_batch_padding(e_cand_to_idxs, e_idx_to_cands, e_adjs, min(search_ment_size, n_ments), min(n_cands, search_entity_size)-1)
                    nega_e = (new_adjs, new_node_cands, new_node_mask)

                    new_scores, _ = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype, ment_ids, ment_mask, desc_ids, desc_mask, None, madj[dc], nega_e, random_new_cand_idx, gold=None, isTrain=False, chosen_ment=small_idxs, isLocal=False)
                    # new_scores: search_ment_size * search_entity_size

                    big_score_in_new, big_idxs_in_new = torch.max(new_scores, dim=1)
                    # big_idxs = random_new_cand_idx[big_idxs_in_new]
                    big_idxs = torch.gather(random_new_cand_idx, 1, big_idxs_in_new.unsqueeze(1)).squeeze(1)
                    final_pde = pde
                    if big_score_in_new.sum() - new_scores[:,0].sum() < 1e-7:
                        death_cnt = death_cnt + 1
                        if death_cnt > death_epoches:
                            if random.random()<0.01:
                                print("DEATH")
                                print("pde:", pde, "old_scores:", cur_scores)
                                print("new_scores:", new_scores)
                                print("------")
                            break
                    else:
#                         print("pde:", pde, "small_idxs:", small_idxs, "cur_cand_idxs[small_idxs]", cur_cand_idxs[small_idxs])
#                         print_flag = True
#                         print("before change old_scores:", cur_scores)
#                         print("big_idxs:", big_idxs)
                        cur_cand_idxs[small_idxs] = big_idxs
                        death_cnt = 0

                pred_ids = cur_cand_idxs
                global_cur_cands = torch.gather(entity_ids, 1, cur_cand_idxs.unsqueeze(1))
                total_ments += n_ments
                total_changed += int((global_cur_cands!=local_cur_cands).sum())
                
            end_time = time.time()
            #if isLocal == False and random.random()<0.1:
            if isLocal == False and final_pde > death_epoches:
                print("dc:", dc, "pred_time:", end_time-start_time, ",pde:", final_pde, ",n_ments:", n_ments)
            if isLocal == False and dc == 0:
                print("global_beta", self.model.global_beta, cur_scores.squeeze(1))
            
            if self.rt_flag:
                self.run_time.append([total_candidates, end_time-start_time])

            pred_entities = [m['selected_cands']['named_cands'][i] if m['selected_cands']['mask'][i] == 1
                             else (m['selected_cands']['named_cands'][0] if m['selected_cands']['mask'][0] == 1 else 'NIL')
                             for (i, m) in zip(pred_ids, batch)]

            doc_names = [m['doc_name'] for m in batch]
            self.added_words = []
            self.added_ents = []
#             if self.seq_len>0 and self.one_entity_once:
#                 predictions[doc_names[-1]].append({'pred': (pred_entities[-1], 0.)})
#             else:
            # for ids in self.model.added_words:
            #     self.added_words.append([self.word_vocab.id2word[idx] for idx in ids])
            # for ids in self.model.added_ents:
            #     self.added_ents.append([self.ent_vocab.id2word[idx] for idx in ids])
            for dname, entity in zip(doc_names, pred_entities):
                predictions[dname].append({'pred': (entity, 0.)})
            #self.record.append(dict({'added_words':self.added_words, 'added_ents':self.added_ents}))
        
        if isLocal==False:
            print("total changed:",total_changed,"total ments:",total_ments,"ratio:",total_changed/total_ments)
        return predictions
    
    def local_only_judge(self, cur_epoch):
        if self.local_cur_stop_epoches >= self.local_stop_epoches:
            return False
        if self.args.use_local_epoches < 0:
            return True
        if cur_epoch >= self.args.use_local_epoches:
            return False
        return True
