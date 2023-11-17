import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from logger import get_missing_parameters_message, get_unexpected_parameters_message

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from pointnet2_utils import PointNetFeaturePropagation
from gpt2_seg import GPT_extractor, GPT_generator
import math
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import numpy as np
from z_order import *


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(
        1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)
        self.knn_2 = KNN(k=1, transpose_mode=True)

    def simplied_morton_sorting(self, xyz, center):
        batch_size, num_points, _ = xyz.shape
        distances_batch = torch.cdist(center, center)
        distances_batch[:, torch.eye(self.num_group).bool()] = float("inf")
        idx_base = torch.arange(
            0, batch_size, device=xyz.device) * self.num_group
        sorted_indices_list = []
        sorted_indices_list.append(idx_base)
        distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
            1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
        distances_batch[idx_base] = float("inf")
        distances_batch = distances_batch.view(
            batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        for i in range(self.num_group - 1):
            distances_batch = distances_batch.view(
                batch_size * self.num_group, self.num_group)
            distances_to_last_batch = distances_batch[sorted_indices_list[-1]]
            closest_point_idx = torch.argmin(distances_to_last_batch, dim=-1)
            closest_point_idx = closest_point_idx + idx_base
            sorted_indices_list.append(closest_point_idx)
            distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
                1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
            distances_batch[closest_point_idx] = float("inf")
            distances_batch = distances_batch.view(
                batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        sorted_indices = torch.stack(sorted_indices_list, dim=-1)
        sorted_indices = sorted_indices.view(-1)
        return sorted_indices

    def morton_sorting(self, xyz, center):
        batch_size, num_points, _ = xyz.shape
        all_indices = []
        for index in range(batch_size):
            points = center[index]
            z = get_z_values(points.cpu().numpy())
            idxs = np.zeros((self.num_group), dtype=np.int32)
            temp = np.arange(self.num_group)
            z_ind = np.argsort(z[temp])
            idxs = temp[z_ind]
            all_indices.append(idxs)
        all_indices = torch.tensor(all_indices, device=xyz.device)

        idx_base = torch.arange(
            0, batch_size, device=xyz.device).view(-1, 1) * self.num_group
        sorted_indices = all_indices + idx_base
        sorted_indices = sorted_indices.view(-1)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(
            0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        # can utilize morton_sorting by choosing morton_sorting function
        sorted_indices = self.simplied_morton_sorting(xyz, center)

        neighborhood = neighborhood.view(
            batch_size * self.num_group, self.group_size, 3)[sorted_indices, :, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3).contiguous()
        center = center.view(
            batch_size * self.num_group, 3)[sorted_indices, :]
        center = center.view(
            batch_size, self.num_group, 3).contiguous()

        return neighborhood, center

class Encoder_small(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)

class Encoder_large(nn.Module):  # Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(
                    drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [7, 15, 23]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t,
                                     2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-
                              1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb


class get_model(nn.Module):
    def __init__(self, cls_dim, trans_dim=384, depth=12, drop_path_rate=0.1, num_heads=6, decoder_depth=4, group_size=32, num_group=128, prop_dim=1024, label_dim1=512, label_dim2=256, encoder_dims=384):
        super().__init__()

        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.cls_dim = cls_dim
        self.num_heads = num_heads

        self.decoder_depth = decoder_depth

        self.group_size = group_size
        self.num_group = num_group

        self.prop_dim = prop_dim

        self.label_dim1 = label_dim1
        self.label_dim2 = label_dim2
        # grouper
        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = encoder_dims
        assert encoder_dims in [384, 768, 1024]
        if encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.blocks = GPT_extractor(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            drop_path_rate=self.drop_path_rate
        )

        self.generator_blocks = GPT_generator(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.decoder_depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            drop_path_rate=self.drop_path_rate
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.2))

        self.propagation_0 = PointNetFeaturePropagation(in_channel=3 * self.encoder_dims + 3,
                                                        mlp=[self.trans_dim * 4, self.prop_dim])

        self.convs1 = nn.Conv1d(6*self.encoder_dims +
                                64 + self.prop_dim, self.label_dim1, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(self.label_dim1, self.label_dim2, 1)
        self.convs3 = nn.Conv1d(self.label_dim2, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(self.label_dim1)
        self.bns2 = nn.BatchNorm1d(self.label_dim2)

        self.relu = nn.ReLU()

        self.loss_func_p1 = ChamferDistanceL1().cuda()
        self.loss_func_p2 = ChamferDistanceL2().cuda()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k,
                         v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('GPT_Transformer'):
                    base_ckpt[k[len('GPT_Transformer.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                    get_missing_parameters_message(incompatible.missing_keys)
                )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                    get_unexpected_parameters_message(
                        incompatible.unexpected_keys)

                )

            print(
                f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, pts, cls_label):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)  # B N 3

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        B = group_input_tokens.shape[0]

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        sos_pos = self.sos_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = torch.cat([sos_pos, pos], dim=1)

        relative_position = center[:, 1:, :] - center[:, :-1, :]
        relative_norm = torch.norm(relative_position, dim=-1, keepdim=True)
        relative_direction = relative_position / relative_norm
        position = torch.cat(
            [center[:, 0, :].unsqueeze(1), relative_direction], dim=1)
        pos_relative = self.pos_embed(position)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        encoded_features, feature_list = self.blocks(x, pos, classify=True)

        encoded_features = torch.cat(
            [encoded_features[:, 0, :].unsqueeze(1), encoded_features[:, 2:-1, :]], dim=1)

        rebuild_points = self.generator_blocks(
            encoded_features, pos_relative)

        neighborhood = neighborhood + center.unsqueeze(2)

        gt_points = neighborhood.reshape(
            B*(self.num_group), self.group_size, 3)

        loss1 = self.loss_func_p1(rebuild_points, gt_points)
        loss2 = self.loss_func_p2(rebuild_points, gt_points)

        feature_list = [self.norm(x).transpose(-1, -2).contiguous()
                        for x in feature_list]
        x = torch.cat(
            (feature_list[0], feature_list[1], feature_list[2]), dim=1)  # 1152
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat(
            (x_max_feature, x_avg_feature, cls_label_feature), 1)  # 1152*2 + 64

        f_level_0 = self.propagation_0(
            pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x)

        x = torch.cat((f_level_0, x_global_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss
