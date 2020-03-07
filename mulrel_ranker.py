import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from GCN_ETHZ.local_ctx_att_ranker import LocalCtxAttRanker
from torch.distributions import Categorical
from GCN_ETHZ.gcn.model import GCN
import GCN_ETHZ.gcn.utils as gcnutil
import copy
import ipdb
import math

np.set_printoptions(threshold=20)


class MulRelRanker(LocalCtxAttRanker):
    """
    multi-relational global model with context token attention, using loopy belief propagation
    """

    def __init__(self, config):

        print('--- create MulRelRanker model ---')
        super(MulRelRanker, self).__init__(config)
        self.dr = config['dr']
        self.gamma = config['gamma']

        self.ent_unk_id = config['entity_voca'].unk_id
        self.word_unk_id = config['word_voca'].unk_id

        self.use_local_epoches = config.get('use_local_epoches', False)
        self.freeze_local = config.get('freeze_local', False)
        self.emb_dims = config['emb_dims']

        # self.cnn used to calculate local scores
        self.cnn = torch.nn.Conv1d(self.emb_dims, 64, kernel_size=3)
        
        # self.gcn = GCN(self.emb_dims, self.emb_dims, self.emb_dims, config['gdr'])
        self.gcn_ment = GCN(64, self.emb_dims, self.emb_dims, config['gdr'])
        self.gcn_entity = GCN(self.emb_dims, self.emb_dims, self.emb_dims, config['gdr'])
        
        # self.cnn_mgraph used to calculate ment embeddings in graph
        # self.cnn_mgraph = torch.nn.Conv1d(self.emb_dims, self.emb_dims, kernel_size=5)
        
        # self.m_e_score = torch.nn.Linear(2 * self.emb_dims, 1)
        self.m_e_diag = torch.nn.Parameter(torch.randn(self.emb_dims))
#         stdv = 1./math.sqrt(self.m_e_diag.size(0))
#         self.m_e_diag.data.uniform_(-stdv, stdv)
#         torch.nn.init.xavier_uniform_(self.m_e_diag)

        self.saved_log_probs = []
        self.rewards = []
        self.actions = []

        self.targets = []
        self.record = False
        if self.freeze_local:
            self.att_mat_diag.requires_grad = False
            self.tok_score_mat_diag.requires_grad = False

        self.param_copy_switch = True

        # Typing feature
        self.type_emb = torch.nn.Parameter(torch.randn([4, 5]))
#         stdv = 1./math.sqrt(self.type_emb.size(1))
#         self.type_emb.data.uniform_(-stdv, stdv)
#         torch.nn.init.xavier_uniform_(self.type_emb)
        
        self.gcned_mat_diag = torch.nn.Parameter(torch.randn(self.emb_dims))
#         stdv = 1./math.sqrt(self.gcned_mat_diag.size(0))
#         self.gcned_mat_diag.data.uniform_(-stdv, stdv)
#         torch.nn.init.xavier_uniform_(self.gcned_mat_diag)
        
        self.global_beta = config['global_beta']
        self.local_score_combine = torch.nn.Sequential(
                torch.nn.Linear(3, 10),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.dr),
                torch.nn.Linear(10, 1))
        self.global_score_combine = torch.nn.Sequential(
                torch.nn.Linear(2, 10),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.dr),
                torch.nn.Linear(10, 1))
#         torch.nn.init.xavier_uniform_(self.local_score_combine[0].weight)
#         torch.nn.init.xavier_uniform_(self.local_score_combine[3].weight)
#         torch.nn.init.xavier_uniform_(self.global_score_combine[0].weight)
#         torch.nn.init.xavier_uniform_(self.global_score_combine[3].weight)

#         self.local_score_combine = torch.nn.Sequential(
#                 torch.nn.Linear(3, 10),
#                 torch.nn.ReLU(),
#                 torch.nn.Dropout(p=self.dr),
#                 torch.nn.Linear(10, 1),
#                 torch.nn.Sigmoid())
#         self.global_score_combine = torch.nn.Sequential(
#                 torch.nn.Linear(2, 10),
#                 torch.nn.ReLU(),
#                 torch.nn.Dropout(p=self.dr),
#                 torch.nn.Linear(10, 1),
#                 torch.nn.Sigmoid())

        self.flag = 0
        self.doc_predict_restore = False

    def get_vec_of_graph(self, graph_embs, node_mask):
        # graph_embs: n_node * emb_dim
        return torch.mean(graph_embs, dim=0)

    def compute_gcned_similarity(self, entity_embs, ment_embs, isTrain=True):
        if isTrain:
            # entity_embs: n_ment * (n_sample+1) * n_node * emb_dim
            # ment_embs: n_ment * emb_dim
            # try to return: n_ment * (n_sample+1)
            n_ment, n_sample, n_node, _ = entity_embs.size()
            assert entity_embs.size(3) == self.emb_dims
            n_sample = n_sample - 1

            msk = torch.zeros(n_ment, n_node).cuda()
            msk[:n_ment, :n_ment] = torch.eye(n_ment)
            msk = msk.unsqueeze(1).unsqueeze(3).repeat(1, n_sample+1, 1, self.emb_dims)
            entity_embs = entity_embs.mul(msk).sum(dim=2)
            # entity_embs: n_ment * (n_sample+1) * emb_dim
            sim_scores = torch.bmm(entity_embs * self.gcned_mat_diag, ment_embs.unsqueeze(2))
            # sim_scores: n_ment * (n_sample+1) * 1
            return sim_scores.squeeze(2)

        else:
            # entity_embs: search_ment_size * search_entity_size * n_node * emb_dim
            # ment_embs: n_ment * emb_dim
            # try to return: search_ment_size * search_entity_size * n_ment
            search_ment_size, search_entity_size, n_node, _ = entity_embs.size()
            n_ment = ment_embs.size(0)
            assert entity_embs.size(3) == self.emb_dims
            assert ment_embs.size(1) == self.emb_dims
            e_embs = entity_embs[:,:,:n_ment,:].reshape(-1, n_ment, self.emb_dims)
            # e_embs: (search_ment_size * search_entity_size) * n_ment * emb_dim
            sim_scores = (e_embs * self.gcned_mat_diag).mul(ment_embs).sum(dim=2)
            return sim_scores.view(search_ment_size, search_entity_size, n_ment)

    def forward(self, token_ids, tok_mask, entity_ids, entity_mask, p_e_m, mtype, etype, ment_ids, ment_mask, desc_ids, desc_mask, m_graph_list, m_graph_adj, nega_e, sample_idx, gold=None, isTrain=True, isLocal=False, chosen_ment=False):

        n_ments, n_cands = entity_ids.size()

        # cnn
        desc_len = desc_ids.size(-1)

        context_len = token_ids.size(-1)

        desc_ids = desc_ids.view(n_ments*n_cands, -1)
        desc_mask = desc_mask.view(n_ments*n_cands, 1, -1)

        context_emb = self.word_embeddings(token_ids)
        desc_emb = self.word_embeddings(desc_ids)

        # context_cnn: n_ment * 1 * 64
        context_cnn = F.max_pool1d(self.cnn(context_emb.permute(0, 2, 1)), context_len-2).permute(0, 2, 1)

        # desc_cnn: n_ment * n_cands * 64
        desc_cnn = F.max_pool1d(self.cnn(desc_emb.permute(0, 2, 1))-(1-desc_mask.float())*1e10, desc_len-2).view(n_ments, n_cands, -1)

        sim = torch.sum(context_cnn*desc_cnn,-1) / torch.sqrt(torch.sum(context_cnn*context_cnn, -1)) / torch.sqrt(torch.sum(desc_cnn*desc_cnn, -1))

        # if not self.oracle:
        #     gold = None

        # Typing feature
        self.mt_emb = torch.matmul(mtype, self.type_emb).view(n_ments, 1, -1)
        self.et_emb = torch.matmul(etype.view(-1, 4), self.type_emb).view(n_ments, n_cands, -1)
        tm = torch.sum(self.mt_emb*self.et_emb, -1, True)

        local_ent_scores = super(MulRelRanker, self).forward(token_ids, tok_mask, entity_ids, entity_mask,p_e_m=None)
        # ment_emb: n_ment * emb_dim (only one graph)
        # ment_emb = F.max_pool1d(self.cnn_mgraph(context_emb.permute(0, 2, 1)), context_len-4).squeeze(2)
        # ment_emb: n_ment * 64 (only one graph)
        ment_emb = context_cnn.squeeze(1)

        local_ent_scores = local_ent_scores.view(n_ments, n_cands)
        p_e_m = torch.log(p_e_m + 1e-20).view(n_ments, n_cands)
        tm = tm.view(n_ments, n_cands)

        if isLocal:
            n_input = n_ments * n_cands
            # padding_zeros = torch.zeros(n_input, 1, requires_grad=False).cuda()
            # inputs = torch.cat([local_ent_scores.view(n_input, -1), p_e_m.view(n_input, -1), tm.view(n_input, -1), padding_zeros, padding_zeros], dim=1)
            # scores = self.score_combine(inputs).view(n_ments, n_cands)
            inputs = torch.cat([local_ent_scores.view(n_input, -1), p_e_m.view(n_input, -1), tm.view(n_input, -1)], dim=1)
            scores = self.local_score_combine(inputs).view(n_ments, n_cands)
            return scores, self.actions
            
#         else:
#             local_ent_scores, tm, ment_emb = self.doc_predict_restore

        nega_adjs, nega_node_cands, nega_node_mask = nega_e
        
        if isTrain or type(chosen_ment) != bool:

            # nega_adjs: n_ment * (n_sample+1) * n_node * n_node
            # nega_node_cands: n_ment * (n_sample+1) * n_node
            # nega_node_mask: n_ment * (n_sample+1) * n_node
            # nega_entity_emb: n_ment * (n_sample+1) * n_node * emb_dim
            nega_entity_emb = self.entity_embeddings(nega_node_cands)

            aaa, bbb, n_node, emb_dim = nega_entity_emb.size()
            ment_emb = gcnutil.feature_norm(ment_emb)
            nega_entity_emb = gcnutil.batch_feature_norm(nega_entity_emb.view(-1, n_node, emb_dim)).view(aaa, bbb, n_node, emb_dim)
            ment_emb = self.gcn_ment(ment_emb, m_graph_adj.long())
            nega_entity_emb = self.gcn_entity.batch_forward(nega_entity_emb.view(-1, n_node, emb_dim), nega_adjs.view(-1, n_node, n_node)).view(aaa, bbb, n_node, emb_dim)

            n_sample = nega_adjs.size(1) - 1

            # mention_graph_emb: emb_dim
            mention_graph_emb = torch.mean(ment_emb, dim=0)
            nega_node_mask2 = nega_node_mask.unsqueeze(3).repeat(1,1,1,self.emb_dims)
            nega_graph_embs = torch.sum(nega_entity_emb.mul(nega_node_mask2), dim=2)
            nega_node_mask2 = torch.sum(nega_node_mask2, dim=2)
            # nega_graph_embs = n_ment * (n_sample+1) * emb_dim
            # nega_entity_emb: n_ment * (n_sample+1) * n_node * emb_dim
            nega_graph_embs = torch.div(nega_graph_embs, nega_node_mask2)

            n_input = nega_graph_embs.size(0) * (n_sample+1)
            mention_graph_emb = mention_graph_emb.unsqueeze(0).repeat(n_input, 1)
            nega_graph_embs = nega_graph_embs.view(n_input, self.emb_dims)
            # mention_graph_emb: n_input * emb_dim
            # nega_graph_embs: n_input * emb_dim
            # graph_scores: (n_ment * (n_sample+1))
            # graph_scores = self.m_e_score(torch.cat([mention_graph_emb, nega_graph_embs], dim=1))
            # graph_scores = torch.cosine_similarity(mention_graph_emb, nega_graph_embs, dim=1)
            graph_scores = mention_graph_emb.mul(self.m_e_diag).mul(nega_graph_embs).sum(dim=1)

            if isTrain:
                # gold: n_ment * 1
                # sample_idx: n_ment * n_sample
                sample_idx2 = torch.cat([sample_idx, gold], dim=1)
                # print("sample_idx2:", sample_idx2)

                sample_local_ent_scores = torch.gather(local_ent_scores, 1, sample_idx2)
                sample_p_e_m = torch.gather(p_e_m, 1, sample_idx2)
                sample_tm = torch.gather(tm, 1, sample_idx2)
                sample_gcnscore = self.compute_gcned_similarity(nega_entity_emb, ment_emb)
                    
                assert sample_local_ent_scores.size() == (n_ments, n_sample+1)
                assert sample_p_e_m.size() == (n_ments, n_sample+1)
                assert sample_tm.size() == (n_ments, n_sample+1)
                assert sample_gcnscore.size() == (n_ments, n_sample+1)

            else:
                # choosing the next-step graph entities while predicting
                #   n_ment == search_ment_size
                #   n_sample+1 == search_entity_size
                n_ments = nega_entity_emb.size(0)
                n_input = n_ments * (n_sample+1)

                # chosen_ment: LongTensor(search_ment_size)
                # sample_idx: search_ment_size * search_entity_size
                sample_local_ent_scores = torch.gather(local_ent_scores[chosen_ment], 1, sample_idx)
                sample_p_e_m = torch.gather(p_e_m[chosen_ment], 1, sample_idx)
                sample_tm = torch.gather(tm[chosen_ment], 1, sample_idx)
                sample_gcnscore = self.compute_gcned_similarity(nega_entity_emb, ment_emb, isTrain=False)[:,:,chosen_ment]
                # sample_gcnscore: search_ment_size * search_entity_size * search_ment_size
                msk = torch.eye(n_ments).cuda().unsqueeze(1).repeat(1,n_sample+1,1)
                # print("msk:", msk.size())
                # print("sample_gcnscore:", sample_gcnscore.size())
                sample_gcnscore = torch.sum(sample_gcnscore * msk, dim=2)

            inputs_local = torch.cat([sample_local_ent_scores.view(n_input, -1), sample_p_e_m.view(n_input, -1), sample_tm.view(n_input, -1)], dim=1)
            inputs_global = torch.cat([graph_scores.view(n_input, -1), sample_gcnscore.view(n_input, -1)], dim=1)
            
            scores_local = self.local_score_combine(inputs_local)
            scores_global = self.global_score_combine(inputs_global)
            
            msk = 1 - 2*(scores_local>1e8)
            scores_local = msk * scores_local
            msk = 1 - 2*(scores_global>1e8)
            scores_global = msk * scores_global
            
            scores = (scores_local*(1-self.global_beta) + self.global_beta*scores_global).view(n_ments, n_sample+1)
            
            return scores, (scores_local.view(n_ments, n_sample+1), scores_global.view(n_ments, n_sample+1))

        else:
            # calculating scores of all current ment-entity pairs
            # to choose some entities to be replaced
            # nega_adjs: n_node * n_node
            # nega_node_cands: n_node
            # nega_node_mask: n_node
            # nega_entity_emb: n_node * emb_dim
            nega_entity_emb = self.entity_embeddings(nega_node_cands)
            ment_emb = gcnutil.feature_norm(ment_emb)
            nega_entity_emb = gcnutil.feature_norm(nega_entity_emb)
            ment_emb = self.gcn_ment(ment_emb, m_graph_adj)
            nega_entity_emb = self.gcn_entity(nega_entity_emb, nega_adjs)

            mention_graph_emb = torch.mean(ment_emb, dim=0)
            entity_graph_emb = torch.mean(nega_entity_emb, dim=0)
            # graph_scores = torch.cosine_similarity(mention_graph_emb, entity_graph_emb, dim=0)
            graph_scores = mention_graph_emb.mul(self.m_e_diag).dot(entity_graph_emb)
            graph_scores = graph_scores * torch.ones(n_ments, 1).cuda()

            sample_idx2 = sample_idx.unsqueeze(1)

            sample_local_ent_scores = torch.gather(local_ent_scores, 1, sample_idx2)
            sample_p_e_m = torch.gather(p_e_m, 1, sample_idx2)
            sample_tm = torch.gather(tm, 1, sample_idx2)

            entity_emb = nega_entity_emb[:n_ments]
            # entity_emb: n_ment * emb_dims
            # ment_emb: n_ment * emb_dims
            sample_gcnscore = torch.mul(ment_emb*self.gcned_mat_diag, entity_emb).sum(dim=1).unsqueeze(1)

            n_input = n_ments
            inputs_local = torch.cat([sample_local_ent_scores, sample_p_e_m, sample_tm], dim=1)
            inputs_global = torch.cat([graph_scores, sample_gcnscore], dim=1)
            scores = (1-self.global_beta) * self.local_score_combine(inputs_local) + self.global_beta * self.global_score_combine(inputs_global)

        # inputs = torch.cat([local_ent_scores.view(n_ments * n_cands, -1),
        #                     torch.log(p_e_m + 1e-20).view(n_ments * n_cands, -1)], dim=1)
        # print("n_ments, n_cands", n_ments, n_cands)
        # print("desc_len, context_len", desc_len, context_len)
        # print("desc_ids", desc_ids.size())
        # print("desc_mask", desc_mask.size())
        # print("input",inputs.size())
        # print("local_ent_scores",local_ent_scores.size())
        # print("p_e_m",p_e_m.size())
        # print("tm",tm.size())
        # print("self.score_combine",self.score_combine)
        
        # assert False
        
#         if torch.isnan(scores).any():
#             ipdb.set_trace()
        return scores, self.actions

    def unique(self, numpy_array):
        t = np.unique(numpy_array)
        return torch.from_numpy(t).type(torch.LongTensor)

    def print_weight_norm(self):
        LocalCtxAttRanker.print_weight_norm(self)

        print('f local - l1.w, b', self.local_score_combine[0].weight.data.norm(), self.local_score_combine[0].bias.data.norm())
        print('f local - l2.w, b', self.local_score_combine[3].weight.data.norm(), self.local_score_combine[3].bias.data.norm())
        print('f global - l1.w, b', self.global_score_combine[0].weight.data.norm(), self.global_score_combine[0].bias.data.norm())
        print('f global - l2.w, b', self.global_score_combine[3].weight.data.norm(), self.global_score_combine[3].bias.data.norm())

    def regularize(self, max_norm=4):
        # super(MulRelRanker, self).regularize(max_norm)
        # print("----MulRelRanker Regularization----")

        l1_w_norm = self.local_score_combine[0].weight.norm()
        l1_b_norm = self.local_score_combine[0].bias.norm()
        l2_w_norm = self.local_score_combine[3].weight.norm()
        l2_b_norm = self.local_score_combine[3].bias.norm()
        
        l1_w_norm2 = self.global_score_combine[0].weight.norm()
        l1_b_norm2 = self.global_score_combine[0].bias.norm()
        l2_w_norm2 = self.global_score_combine[3].weight.norm()
        l2_b_norm2 = self.global_score_combine[3].bias.norm()

        if (l1_w_norm > max_norm).data.all():
            self.local_score_combine[0].weight.data = self.local_score_combine[0].weight.data * max_norm / l1_w_norm.data
        if (l1_b_norm > max_norm).data.all():
            self.local_score_combine[0].bias.data = self.local_score_combine[0].bias.data * max_norm / l1_b_norm.data
        if (l2_w_norm > max_norm).data.all():
            self.local_score_combine[3].weight.data = self.local_score_combine[3].weight.data * max_norm / l2_w_norm.data
        if (l2_b_norm > max_norm).data.all():
            self.local_score_combine[3].bias.data = self.local_score_combine[3].bias.data * max_norm / l2_b_norm.data
        if (l1_w_norm2 > max_norm).data.all():
            self.global_score_combine[0].weight.data = self.global_score_combine[0].weight.data * max_norm / l1_w_norm2.data
        if (l1_b_norm2 > max_norm).data.all():
            self.global_score_combine[0].bias.data = self.global_score_combine[0].bias.data * max_norm / l1_b_norm2.data
        if (l2_w_norm2 > max_norm).data.all():
            self.global_score_combine[3].weight.data = self.global_score_combine[3].weight.data * max_norm / l2_w_norm2.data
        if (l2_b_norm2 > max_norm).data.all():
            self.global_score_combine[3].bias.data = self.global_score_combine[3].bias.data * max_norm / l2_b_norm2.data

    def finish_episode(self, rewards_arr, log_prob_arr):
        if len(rewards_arr) != len(log_prob_arr):
            print("Size mismatch between Rwards and Log_probs!")
            return

        policy_loss = []
        rewards = []

        # we only give a non-zero reward when done
        g_return = sum(rewards_arr) / len(rewards_arr)

        # add the final return in the last step
        rewards.insert(0, g_return)

        R = g_return
        for idx in range(len(rewards_arr) - 1):
            R = R * self.gamma
            rewards.insert(0, R)

        rewards = torch.from_numpy(np.array(rewards)).type(torch.cuda.FloatTensor)

        for log_prob, reward in zip(log_prob_arr, rewards):
            policy_loss.append(-log_prob * reward)

        policy_loss = torch.cat(policy_loss).sum()

        return policy_loss

    def loss(self, scores, true_pos, method="SL", lamb=1e-7, isLocal=False):
        #loss = None

        # print("----MulRelRanker Loss----")
#         if method == "SL":
        if isLocal:
            loss = F.multi_margin_loss(scores, true_pos, margin=self.margin)
        else:
            loss = F.multi_margin_loss(scores, true_pos, margin=self.margin_global)
#         elif method == "RL":
#             loss = self.finish_episode(self.rewards, self.saved_log_probs)

        return loss
    
    def local_parameter(self):
        lst = [super(MulRelRanker, self).parameters(), self.cnn.parameters(), self.local_score_combine.parameters()]
        return [p for a in lst for p in a if p.requires_grad] + [self.type_emb]
        
    def global_parameter(self):
        # lst = [self.gcn.parameters(), self.cnn_mgraph.parameters(), self.m_e_score.parameters()]
        # lst = [self.gcn.parameters(), self.cnn_mgraph.parameters(), self.global_score_combine.parameters()]
        lst = [self.gcn_ment.parameters(), self.gcn_entity.parameters(), self.global_score_combine.parameters()]
        return [p for a in lst for p in a if p.requires_grad] + [self.gcned_mat_diag, self.m_e_diag]
