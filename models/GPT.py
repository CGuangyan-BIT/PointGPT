import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, attn_mask):

        x = self.ln_1(x)
        # a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT_extractor(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, num_classes, trans_dim, group_size
    ):
        super(GPT_extractor, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*(self.group_size), 1)
        )

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.cls_norm = nn.LayerNorm(self.trans_dim)

    def forward(self, h, pos, attn_mask, classify=False):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        batch, length, C = h.shape

        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)

        # prepend sos token
        sos = torch.ones(1, batch, self.embed_dim, device=h.device) * self.sos
        if not classify:
            h = torch.cat([sos, h[:-1, :, :]], axis=0)
        else:
            h = torch.cat([sos, h], axis=0)

        # transformer
        for layer in self.layers:
            h = layer(h + pos, attn_mask)

        h = self.ln_f(h)

        encoded_points = h.transpose(0, 1)
        if not classify:
            return encoded_points

        h = h.transpose(0, 1)
        h = self.cls_norm(h)
        concat_f = torch.cat([h[:, 1], h[:, 2:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret, encoded_points


class GPT_generator(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, trans_dim, group_size
    ):
        super(GPT_generator, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*(self.group_size), 1)
        )

    def forward(self, h, pos, attn_mask):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        batch, length, C = h.shape

        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)

        # transformer
        for layer in self.layers:
            h = layer(h + pos, attn_mask)

        h = self.ln_f(h)

        rebuild_points = self.increase_dim(h.transpose(1, 2)).transpose(
            1, 2).transpose(0, 1).reshape(batch * length, -1, 3)

        return rebuild_points
