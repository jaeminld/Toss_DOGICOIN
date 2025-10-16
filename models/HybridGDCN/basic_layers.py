# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""
import numpy as np
import torch
import torch.nn as nn


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim, concat=False):
        super().__init__()
        if isinstance(embed_dim, int):
            self.feature_embs = [embed_dim] * len(field_dims)
        elif isinstance(embed_dim, list):
            self.feature_embs = embed_dim
        self.feature_nums = field_dims
        self.embed_dict = nn.ModuleDict()
        self.feature_sum = sum(self.feature_nums)
        self.emb_sum = sum(self.feature_embs)
        self.len_field = len(self.feature_nums)
        self.concat = concat
        for field_index, (feature_num, feature_emb) in enumerate(zip(self.feature_nums, self.feature_embs)):
            embed = torch.nn.Embedding(feature_num, feature_emb)
            torch.nn.init.xavier_uniform_(embed.weight)
            self.embed_dict[str(field_index)] = embed

    def forward(self, x):
        sparse_embs = [self.embed_dict[str(index)](x[:, index]) for index in range(self.len_field)]

        if self.concat:
            # Independent dimension for DCN、DCN-V2 and GDCN
            sparse_embs = torch.cat(sparse_embs, dim=1)
            return sparse_embs
        # Equal dimension for most models
        sparse_embs = torch.stack(sparse_embs, dim=1)
        return sparse_embs


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.mlp(x)