import torch
import torch.nn as nn
import torch.nn.functional as F
import GCN_ETHZ.utils as utils
from GCN_ETHZ.abstract_word_entity import AbstractWordEntity
import copy
import math

class LocalCtxAttRanker(AbstractWordEntity):
    """
    local model with context token attention (from G&H's EMNLP paper)
    """

    def __init__(self, config):
        print('--- create LocalCtxAttRanker model ---')

        config['word_embeddings_class'] = nn.Embedding
        config['entity_embeddings_class'] = nn.Embedding
        super(LocalCtxAttRanker, self).__init__(config)

        self.hid_dims = config['hid_dims']
        self.tok_top_n = config['tok_top_n']
        self.margin = config['margin']
        self.margin_global = config['margin_global']

        self.att_mat_diag = nn.Parameter(torch.zeros(self.emb_dims))
        self.tok_score_mat_diag = nn.Parameter(torch.zeros(self.emb_dims))
#         stdv = 1./math.sqrt(self.emb_dims)
#         self.att_mat_diag.data.uniform_(-stdv, stdv)
#         self.tok_score_mat_diag.data.uniform_(-stdv, stdv)
        
        self.local_ctx_dr = nn.Dropout(p=0)

        self.ment_att_mat_diag = nn.Parameter(torch.zeros(self.emb_dims))
        self.ment_score_mat_diag = nn.Parameter(torch.zeros(self.emb_dims))

        self.ment_att_mat_diag.requires_grad = False
        self.ment_score_mat_diag.requires_grad = False

        self.param_copy_switch = True

    def forward(self, token_ids, tok_mask, entity_ids, entity_mask, p_e_m=None):
        batchsize, n_words = token_ids.size()
        n_entities = entity_ids.size(1)
        tok_mask = tok_mask.view(batchsize, 1, -1)

        tok_vecs = self.word_embeddings(token_ids)
        entity_vecs = self.entity_embeddings(entity_ids)

        # att
        ent_tok_att_scores = torch.bmm(entity_vecs * self.att_mat_diag, tok_vecs.permute(0, 2, 1))
        ent_tok_att_scores = (ent_tok_att_scores * tok_mask).add_((tok_mask - 1).mul_(50))
        # ent_tok_att_scores: n_ment * n_cand * ctxlen
        tok_att_scores, _ = torch.max(ent_tok_att_scores, dim=1)
        # ent_tok_att_scores: n_ment * ctxlen
        top_tok_att_scores, top_tok_att_ids = torch.topk(tok_att_scores, dim=1, k=min(self.tok_top_n, n_words))
        # top_tok_att_scores: n_ment * k
        # top_tok_att_ids: n_ment * k, values in 0-ctxlen
        att_probs = F.softmax(top_tok_att_scores, dim=1).view(batchsize, -1, 1)
        # att_probs: n_ment * k * 1
        att_probs = att_probs / torch.sum(att_probs, dim=1, keepdim=True)
        
        tok_att_scores_nan = torch.ones_like(tok_att_scores, requires_grad=False).cuda()
        tok_att_scores_nan[torch.isnan(tok_att_scores)] = 0.
        tok_att_scores = tok_att_scores.mul(tok_att_scores_nan)
        
        selected_tok_vecs = torch.gather(tok_vecs, dim=1,
                                         index=top_tok_att_ids.view(batchsize, -1, 1).repeat(1, 1, tok_vecs.size(2)))
        # selected_tok_vecs: n_ment * k * dim
        ctx_vecs = torch.sum((selected_tok_vecs * self.tok_score_mat_diag) * att_probs, dim=1, keepdim=True)
        ctx_vecs = self.local_ctx_dr(ctx_vecs)
        ent_ctx_scores = torch.bmm(entity_vecs, ctx_vecs.permute(0, 2, 1)).view(batchsize, n_entities)

        scores = ent_ctx_scores
        scores = (scores * entity_mask).add_((entity_mask - 1).mul_(1e10))

        return scores

    def compute_local_similarity(self, token_ids, tok_mask, entity_ids, entity_mask):
        if self.param_copy_switch:
            self.ment_att_mat_diag = copy.deepcopy(self.att_mat_diag)
            self.ment_score_mat_diag = copy.deepcopy(self.tok_score_mat_diag)

            self.ment_att_mat_diag.requires_grad = False
            self.ment_score_mat_diag.requires_grad = False

            self.param_copy_switch = False

        batchsize, n_words = token_ids.size()
        n_entities = entity_ids.size(1)
        tok_mask = tok_mask.view(batchsize, 1, -1)

        tok_vecs = self.word_embeddings(token_ids)
        entity_vecs = self.word_embeddings(entity_ids)

        # att
        ent_tok_att_scores = torch.bmm(entity_vecs * self.ment_att_mat_diag, tok_vecs.permute(0, 2, 1))
        ent_tok_att_scores = (ent_tok_att_scores * tok_mask).add_((tok_mask - 1).mul_(1e10))
        tok_att_scores, _ = torch.max(ent_tok_att_scores, dim=1)
        top_tok_att_scores, top_tok_att_ids = torch.topk(tok_att_scores, dim=1, k=min(self.tok_top_n, n_words))
        att_probs = F.softmax(top_tok_att_scores, dim=1).view(batchsize, -1, 1)
        att_probs = att_probs / torch.sum(att_probs, dim=1, keepdim=True)

        selected_tok_vecs = torch.gather(tok_vecs, dim=1,
                                         index=top_tok_att_ids.view(batchsize, -1, 1).repeat(1, 1, tok_vecs.size(2)))
        ctx_vecs = torch.sum((selected_tok_vecs * self.ment_score_mat_diag) * att_probs, dim=1, keepdim=True)
        ctx_vecs = self.local_ctx_dr(ctx_vecs)
        ent_ctx_scores = torch.bmm(entity_vecs, ctx_vecs.permute(0, 2, 1)).view(batchsize, n_entities)

        scores = ent_ctx_scores

        scores = scores * entity_mask

        return scores

    def print_weight_norm(self):
        print('att_mat_diag', self.att_mat_diag.data.norm())
        print('tok_score_mat_diag', self.tok_score_mat_diag.data.norm())

        print('ment_att_mat_diag', self.ment_att_mat_diag.data.norm())
        print('ment_score_mat_diag', self.ment_score_mat_diag.data.norm())

