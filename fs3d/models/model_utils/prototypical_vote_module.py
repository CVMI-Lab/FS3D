# Copyright (c) OpenMMLab. All rights reserved.

from mmcv import is_tuple_of
from mmcv.cnn import ConvModule
from torch import nn as nn
from fs3d.models.builder import build_loss
import torch
from mmcv.cnn import xavier_init
from mmcv.cnn.utils import constant_init, kaiming_init
from functools import partial
from mmcv.cnn.bricks.norm import build_norm_layer

class PrototypicalVoteModule(nn.Module):

    def __init__(self,
                 in_channels,
                 vote_per_seed=1,
                 gt_per_seed=3,
                 num_points=-1,
                 conv_channels=(16, 16),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 norm_feats=True,
                 with_res_feat=True,
                 vote_xyz_range=None,
                 vote_loss=None,
                 use_refine=None,
                 vote_feature_loss=None,
                 transformer_nhead=4,
                 transformer_dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.vote_per_seed = vote_per_seed
        self.gt_per_seed = gt_per_seed
        self.num_points = num_points
        self.norm_feats = norm_feats
        self.with_res_feat = with_res_feat

        assert vote_xyz_range is None or is_tuple_of(vote_xyz_range, float)
        self.vote_xyz_range = vote_xyz_range

        if vote_loss is not None:
            self.vote_loss = build_loss(vote_loss)

        if vote_feature_loss is not None:
            self.vote_feature_loss = build_loss(vote_feature_loss)

        self.point_refinement = TransformerLayer(
            d_model=self.in_channels,
            nhead=transformer_nhead,
            dropout=transformer_dropout,
        )
        prev_channels = in_channels

        if with_res_feat:
            out_channel = (3 + in_channels) * self.vote_per_seed
        else:
            out_channel = 3 * self.vote_per_seed
        self.vote_layer = nn.Conv1d(prev_channels, out_channel, 1)

    def forward(self, seed_points, seed_feats, prototypes=None, stage="train", gt_vote=None):

        if self.num_points != -1:
            assert self.num_points < seed_points.shape[1], \
                f'Number of vote points ({self.num_points}) should be '\
                f'smaller than seed points size ({seed_points.shape[1]})'
            seed_points = seed_points[:, :self.num_points]
            seed_feats = seed_feats[..., :self.num_points]

        batch_size, feat_channels, num_seed = seed_feats.shape
        num_vote = num_seed * self.vote_per_seed

        x = self.point_refinement(seed_feats, xyz=seed_points, prototypes=prototypes, stage=stage)[0]

        # (batch_size, (3+out_dim)*vote_per_seed, num_seed)
        middle_feature = seed_feats
        votes = self.vote_layer(x)

        votes = votes.transpose(2, 1).view(batch_size, num_seed,
                                           self.vote_per_seed, -1)
        offset = votes[:, :, :, 0:3]

        if self.vote_xyz_range is not None:
            # print("self.vote_xyz_range")
            limited_offset_list = []
            for axis in range(len(self.vote_xyz_range)):
                limited_offset_list.append(offset[..., axis].clamp(
                    min=-self.vote_xyz_range[axis],
                    max=self.vote_xyz_range[axis]))
            limited_offset = torch.stack(limited_offset_list, -1)
            vote_points = (seed_points.unsqueeze(2) +
                           limited_offset).contiguous()
        else:
            vote_points = (seed_points.unsqueeze(2) + offset).contiguous()

        vote_points = vote_points.view(batch_size, num_vote, 3)
        offset = offset.reshape(batch_size, num_vote, 3).transpose(2, 1)

        if self.with_res_feat:
            res_feats = votes[:, :, :, 3:]
            vote_feats = (seed_feats.transpose(2, 1).unsqueeze(2) +
                          res_feats).contiguous()
            vote_feats = vote_feats.view(batch_size,
                                         num_vote, feat_channels).transpose(
                                             2, 1).contiguous()
            if self.norm_feats:
                "t"
                features_norm = torch.norm(vote_feats, p=2, dim=1)
                vote_feats = vote_feats.div(features_norm.unsqueeze(1))
        else:
            vote_feats = seed_feats

        return vote_points, vote_feats, offset, middle_feature, seed_feats

    def get_loss(self, seed_points, vote_points, seed_indices,
                 vote_targets_mask, vote_targets):

        batch_size, num_seed = seed_points.shape[:2]

        seed_gt_votes_mask = torch.gather(vote_targets_mask, 1,
                                          seed_indices).float()

        seed_indices_expand = seed_indices.unsqueeze(-1).repeat(
            1, 1, 3 * self.gt_per_seed)
        seed_gt_votes = torch.gather(vote_targets, 1, seed_indices_expand)
        seed_gt_votes += seed_points.repeat(1, 1, self.gt_per_seed)

        weight = seed_gt_votes_mask / (torch.sum(seed_gt_votes_mask) + 1e-6)

        distance = self.vote_loss(
            vote_points.view(batch_size * num_seed, -1, 3),
            seed_gt_votes.view(batch_size * num_seed, -1, 3),
            dst_weight=weight.view(batch_size * num_seed, 1))[1]
        vote_loss = torch.sum(torch.min(distance, dim=1)[0])

        return vote_loss

class TransformerLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dropout=0.1, embedding_num=120):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.embedding_num = embedding_num

        self.memory_banks_embedding = torch.nn.Embedding(self.embedding_num, d_model).cuda()
        self.memory_banks = self.memory_banks_embedding.weight

        _, self.norm_last = build_norm_layer(dict(type='BN1d'), d_model)
        self.d_model = d_model

        self.mu = 0.999
        self.prototype_num_per_class = 10

        self.linear_1 = ConvModule(
            d_model,
            d_model,
            1,
            padding=0,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'),
            bias=True,
            inplace=True)

        self.linear_2 = ConvModule(
            d_model,
            d_model,
            1,
            padding=0,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'),
            bias=True,
            inplace=True)

        # self.init_weights()

    def init_weights(self):
        # initialize transformer
        for m in self.parameters():
            if m.dim() > 1:
                xavier_init(m, distribution='uniform')
        constant_init(self.norm_last, 1, bias=0)
        self.memory_banks_embedding.weight[:] = torch.rand(self.embedding_num, self.d_model)


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def update_memory(self, features, info):

        features = features.permute(1, 0, 2).detach()
        indices = info['fp_indices']
        pts_semantic_masks = info['pts_semantic_mask']
        pts_instance_masks = info['pts_instance_mask']

        feature_list = [[] for i in range(self.memory_banks.shape[0])]

        for batch_index in range(features.shape[0]):
            feature = features[batch_index]
            indice = indices[batch_index]
            pts_semantic_mask = pts_semantic_masks[batch_index][indice]
            pts_instance_mask = pts_instance_masks[batch_index][indice]

            foreground_indice = torch.nonzero(pts_semantic_mask < 18, as_tuple=False).squeeze(-1)
            foreground_features = feature[foreground_indice]
            foreground_semantic_mask = pts_semantic_mask[foreground_indice]
            similarity_matrix = torch.mm(foreground_features, self.memory_banks.permute(1, 0))

            max_indices = torch.argmax(similarity_matrix, dim=1)
            max_all_indices = torch.sort(similarity_matrix, dim=1, descending=True)[1]

            feature_list_count = torch.zeros(len(feature_list))
            class_list_count = torch.zeros(18)

            contain_num = int(max_all_indices.shape[0] / max_all_indices.shape[1]) + 1

            for i in range(len(max_indices)):

                label = foreground_semantic_mask[i]
                if class_list_count[label] >= self.prototype_num_per_class:
                    continue
                class_list_count[label] += 1

                max_all_indice = max_all_indices[i]
                indice_in_indice = torch.nonzero(feature_list_count[max_all_indice] < contain_num, as_tuple=False)[0]
                max_indice = max_all_indice[indice_in_indice]
                feature_list_count[max_indice] += 1
                foreground_feature = foreground_features[i].unsqueeze(0)
                feature_list[max_indice].append(foreground_feature)

        for index in range(len(feature_list)):

            if len(feature_list[index]) == 0:
                continue

            one_slot = torch.cat(feature_list[index], 0)
            one_slot = torch.mean(one_slot, dim=0)
            features_norm = torch.norm(one_slot, p=2, dim=0)
            one_slot = one_slot.div(features_norm)
            self.memory_banks[index] = self.mu * self.memory_banks[index] + (1 - self.mu) * one_slot

        features_norm = torch.norm(self.memory_banks_embedding.weight, p=2, dim=1).unsqueeze(1)
        self.memory_banks_embedding.weight[:] = self.memory_banks_embedding.weight.div(features_norm)

    def feature_norm(self, feature):
        features_norm = torch.norm(feature, p=2, dim=1)
        feature = feature.div(features_norm.unsqueeze(1))
        return feature

    def forward_pre(self, tgt, memory=None,
                    tgt_mask = None,
                    memory_mask = None,
                    tgt_key_padding_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None,
                    return_attn_weights = False,
                    xyz = None,
                    prototypes=None,
                    stage="train"):

        tgt = tgt.permute(2, 0, 1)
        q = tgt

        k = v = self.memory_banks.unsqueeze(0).repeat(tgt.shape[1], 1, 1).permute(1, 0, 2)

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        if isinstance(prototypes, dict):
            self.update_memory(q, prototypes)

        tgt = tgt + self.dropout1(tgt2)

        tgt = tgt.permute(1, 2, 0)

        tgt = self.norm_last(tgt)

        tgt = self.linear_1(tgt)

        tgt = self.linear_2(tgt)

        return tgt, None

    def forward(self, tgt, memory=None,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None,
                return_attn_weights = False,
                xyz=None,
                prototypes=None,
                stage="train"):

        return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights, xyz=xyz, prototypes=prototypes, stage=stage)

class BatchNormDim1Swap(nn.BatchNorm1d):
    """
    Used for nn.Transformer that uses a HW x N x C rep
    """

    def forward(self, x):
        """
        x: HW x N x C
        permute to N x C x HW
        Apply BN on C
        permute back
        """
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super(BatchNormDim1Swap, self).forward(x)
        # x: n x c x hw -> hw x n x c
        x = x.permute(2, 0, 1)
        return x


NORM_DICT = {
    "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}

