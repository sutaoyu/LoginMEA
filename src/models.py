#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch_scatter import scatter_sum

from timm.models.layers import Mlp, DropPath

try:
    from layers import *
except:
    from src.layers import *


class MrFusionGraphAttention(nn.Module):
    def __init__(
        self, node_size, rel_size, triple_size, node_dim, depth=1, mr_fusion_type=0
    ):
        super(MrFusionGraphAttention, self).__init__()

        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.node_dim = node_dim
        self.activation = torch.nn.Tanh()
        self.depth = depth
        self.attn_kernels = nn.ParameterList()
        self.fusion_type = mr_fusion_type

        for l in range(self.depth):
            attn_kernel = torch.nn.Parameter(
                data=torch.empty(self.node_dim, 1, dtype=torch.float32)
            )
            nn.init.xavier_uniform_(attn_kernel)
            self.attn_kernels.append(attn_kernel)
        self.modal_num = 3
        self.emb_atten_cat = MultiModalFusion(self.modal_num)
        if self.fusion_type == 7:
            self.fusion = AdaptiveLMFusion(
                self.modal_num,
                [300, 300, 300],
                output_dim=300,
                rank=8,
                use_softmax=False,
            )

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj = inputs[2]
        r_index = inputs[3]
        r_val = inputs[4]

        if self.fusion_type > 0 and len(inputs) > 5:
            embs = inputs[5]
            early_features = self.emb_atten_cat(embs)
            outputs.append(self.activation(early_features))

            after_features = self.fusion(embs)
            features = self.activation(after_features)
        else:
            features = self.activation(features)
            outputs.append(features)

        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]
            tri_rel = torch.sparse_coo_tensor(
                indices=r_index,
                values=r_val,
                size=[self.triple_size, self.rel_size],
                dtype=torch.float32,
            )
            tri_rel = torch.sparse.mm(tri_rel, rel_emb)
            neighs = features[adj[1, :].long()]
            tri_rel = F.normalize(tri_rel, dim=1, p=2)

            neighs = (
                neighs - 2 * torch.sum(neighs * tri_rel, dim=1, keepdim=True) * tri_rel
            )

            att = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)
            att = torch.sparse_coo_tensor(
                indices=adj, values=att, size=[self.node_size, self.node_size]
            )
            att = torch.sparse.softmax(att, dim=1)

            new_features = scatter_sum(
                src=neighs * torch.unsqueeze(att.coalesce().values(), dim=-1),
                dim=0,
                index=adj[0, :].long(),
            )
            features = self.activation(new_features)
            outputs.append(features)
        outputs = torch.cat(outputs, dim=-1)
        final_outputs = outputs
        return final_outputs


class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(
            torch.ones((self.modal_num, 1)), requires_grad=self.requires_grad
        )

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [
            weight_norm[idx] * F.normalize(embs[idx])
            for idx in range(self.modal_num)
            if embs[idx] is not None
        ]
        joint_emb = torch.cat(embs, dim=1)
        return joint_emb


class AdaptiveLMFusion(nn.Module):
    """
    Adaptive Low-rank Multimodal Fusion
    """

    def __init__(
        self,
        modal_num,
        input_dim_list,
        output_dim=300,
        rank=8,
        use_softmax=False,
        fusion_beta=0.5,
        use_layernorm=False,
    ):
        super(AdaptiveLMFusion, self).__init__()

        self.modal_num = modal_num
        self.rank = rank
        self.use_softmax = use_softmax
        self.output_dim = input_dim_list[0]
        self.use_layernorm = use_layernorm
        self.fusion_beta = fusion_beta

        self.ent_attn = nn.Linear(input_dim_list[0], 1, bias=False)
        self.ent_attn.requires_grad_(True)

        self.fusion_beta = fusion_beta

        self.factor_list = nn.ParameterList()
        for i in range(self.modal_num):
            input_dim = input_dim_list[i]
            factor = nn.Parameter(
                torch.Tensor(self.rank, input_dim + 1, self.output_dim)
            )
            nn.init.xavier_normal_(factor)
            self.factor_list.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(self.output_dim)

    def forward(self, embs):
        """
        Args:
            img_embed: tensor of shape (batch_size, img_dim)
            relation_embed: tensor of shape (batch_size, relation_dim)
            attribute_embed: tensor of shape (batch_size, attribute_dim)
        """
        assert len(embs) == self.modal_num

        device = embs[0].device
        batch_size = embs[0].size(0)

        emb_e = torch.stack(embs, dim=1)
        emb_u = torch.tanh(emb_e)
        scores = self.ent_attn(emb_u).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=-1)

        fusion_zy = 1.0
        for idx in range(self.modal_num):
            emb = F.normalize(embs[idx]) * attention_weights[:, idx].view(-1, 1)
            emb_h = torch.cat((torch.ones(batch_size, 1, device=device), emb), dim=1)
            factor = self.factor_list[idx]
            emb_h = torch.matmul(emb_h, factor)
            fusion_zy = fusion_zy * emb_h
        lmf_output = (
            torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze()
            + self.fusion_bias
        )

        lmf_output = lmf_output.view(-1, self.output_dim)
        if self.use_softmax:
            lmf_output = torch.softmax(lmf_output, dim=-1)
        if self.use_layernorm:
            lmf_output = self.layernorm(lmf_output)
        output = lmf_output
        return output


class MultiModalEncoderMrFusion(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units

    """

    def __init__(
        self,
        args,
        ent_num,
        rel_size,
        triple_size,
        img_feature_dim,
        adj_matrix,
        r_index,
        r_val,
        rel_matrix,
        ent_matrix,
        char_feature_dim=None,
        use_project_head=False,
        left_ents=None,
        right_ents=None,
    ):
        super(MultiModalEncoderMrFusion, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(1000, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)

        self.left_ents = left_ents
        self.right_ents = right_ents
        if "Dualmodal" in self.args.structure_encoder:
            self.node_hidden = self.input_dim
            self.rel_hidden = self.input_dim
            self.node_size = ent_num
            self.rel_size = rel_size
            self.triple_size = triple_size
            self.depth = 2
            self.adj_list = adj_matrix
            self.r_index = r_index
            self.r_val = r_val
            self.rel_adj = rel_matrix
            self.ent_adj = ent_matrix
            self.dropout = nn.Dropout(args.dropout)

            self.mr_fusion_type = self.args.mr_fusion_type
            self.final_fusion_type = self.args.final_fusion_type

            self.ent_embedding = nn.Embedding(self.node_size, self.node_hidden)
            self.rel_embedding = nn.Embedding(self.rel_size, self.rel_hidden)
            self.rel_fc_dual = nn.Linear(self.rel_size, self.rel_hidden)
            nn.init.xavier_uniform_(self.ent_embedding.weight)
            nn.init.xavier_uniform_(self.rel_embedding.weight)

            self.e_encoder = MrFusionGraphAttention(
                node_size=self.node_size,
                rel_size=self.rel_size,
                triple_size=self.triple_size,
                node_dim=self.node_hidden,
                depth=self.depth,
                mr_fusion_type=self.mr_fusion_type,
            )
            self.joint_encoder = MrFusionGraphAttention(
                node_size=self.node_size,
                rel_size=self.rel_size,
                triple_size=self.triple_size,
                node_dim=self.node_hidden,
                depth=self.depth,
                mr_fusion_type=self.mr_fusion_type,
            )
        if self.final_fusion_type == 2:
            self.fusion = MultiModalFusion(
                modal_num=2, with_weight=self.args.with_weight
            )

    def avg(self, adj, emb, size: int):
        adj = torch.sparse_coo_tensor(
            indices=adj,
            values=torch.ones_like(adj[0, :], dtype=torch.float32),
            size=[self.node_size, size],
        )
        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)

    def forward(
        self,
        input_idx,
        adj,
        img_features=None,
        rel_features=None,
        att_features=None,
        name_features=None,
        char_features=None,
    ):

        if self.args.w_img:
            img_emb = self.img_fc(img_features)
        else:
            img_emb = None
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None

        if self.args.w_gcn:
            if self.args.structure_encoder == "Dualmodal-joint-Mr":
                joint_emb = []
                if img_emb is not None:
                    joint_emb.append(img_emb)
                if rel_emb is not None:
                    joint_emb.append(rel_emb)
                if att_emb is not None:
                    joint_emb.append(att_emb)

                opt = [
                    self.rel_embedding.weight,
                    self.adj_list,
                    self.r_index,
                    self.r_val,
                ]
                ent_feature = self.avg(
                    self.ent_adj, self.ent_embedding.weight, self.node_size
                )
                gph_emb = self.e_encoder([ent_feature] + opt)

                if self.final_fusion_type == -1:
                    joint_emb = gph_emb
                else:
                    joint_emb = self.joint_encoder([None] + opt + [joint_emb])
                    joint_emb = torch.cat([gph_emb, joint_emb], dim=-1)
        else:
            gph_emb = None
        name_emb, char_emb = None, None
        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb
