import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Modified from https://github.com/yabufarha/ms-tcn/edit/master/model.py
class SingleStageTCN(nn.Module):
    def __init__(self, num_layers, input_dim, middle_dim, output_dim, dropout=0.0):
        super(SingleStageTCN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.conv_in = nn.Conv1d(input_dim, middle_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, middle_dim, middle_dim)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(middle_dim, output_dim, 1)

    def forward(self, x):  # Input: (N, F, T)   Output: (N, F, T)
        x = self.dropout(x)
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


# Modified from https://github.com/yabufarha/ms-tcn/edit/master/model.py
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, input_dim, output_dim):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(input_dim, output_dim, 3,
                                      padding=dilation, dilation=dilation, padding_mode='replicate')
        self.conv_out = nn.Conv1d(output_dim, output_dim, 1)

    def forward(self, x):  # Input: (N, F, T)   Output: (N, F, T)
        out = F.relu(self.conv_dilated(x))
        out = self.conv_out(out)
        return x + out


class EmbeddingModule(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(EmbeddingModule, self).__init__()

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_1 = nn.Conv1d(input_dim, output_dim, 1)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv1d(output_dim, output_dim, 1)

    def forward(self, x):  # Input: (N, F, T)   Output: (N, F, T)

        x = self.dropout(x.unsqueeze(3)).squeeze(3)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)

        return x


class PreparationModule(nn.Module):  # Be careful that whether this module is in the optimizer

    def __init__(self, input_dim_list, embedding_dim_list, instance_norm_flags):

        super(PreparationModule, self).__init__()

        self.num_feature_types = len(input_dim_list)

        self.prepared_dim_list = [embedding_dim_list[i]
                                  if embedding_dim_list[i] else input_dim_list[i]
                                  for i in range(self.num_feature_types)]

        self.instance_norm = [nn.InstanceNorm1d(input_dim_list[i], track_running_stats=False)
                              if instance_norm_flags[i] else None
                              for i in range(self.num_feature_types)]

        self.instance_norm = nn.ModuleList(self.instance_norm)

        dropout_rate_list = []
        for i in range(self.num_feature_types):
            if input_dim_list[i] >= 1000:
                dropout_rate_list.append(0.9)
            elif input_dim_list[i] >= 100:
                dropout_rate_list.append(0.5)
            else:
                dropout_rate_list.append(0.0)

        self.embedding = [EmbeddingModule(input_dim_list[i], embedding_dim_list[i], dropout_rate_list[i])
                          if embedding_dim_list[i] else None
                          for i in range(self.num_feature_types)]

        self.embedding = nn.ModuleList(self.embedding)

    def get_total_dim(self):
        return np.array(self.prepared_dim_list).sum()

    def get_prepared_dim(self):
        return self.prepared_dim_list

    def forward(self, x):  # Input: [(N, F, T)]   Output: [(N, F, T)]

        x = [self.instance_norm[i](x[i]) if self.instance_norm[i] else x[i]
             for i in range(self.num_feature_types)]

        x = [self.embedding[i](x[i]) if self.embedding[i] else x[i]
             for i in range(self.num_feature_types)]

        return x


class ChannelMLPModule(nn.Module):

    def __init__(self, input_dim, middle_dim, output_dim, num_layers):
        super(ChannelMLPModule, self).__init__()

        self.layers = []

        assert (num_layers >= 2)

        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Conv1d(input_dim, middle_dim, 1))
                self.layers.append(nn.ReLU())
            elif i == num_layers - 1:
                self.layers.append(nn.Conv1d(middle_dim, output_dim, 1))
            else:
                self.layers.append(nn.Conv1d(middle_dim, middle_dim, 1))
                self.layers.append(nn.ReLU())

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):  # Input: (N, F, T)   Output: (N, F, T)

        for layer in self.layers:
            x = layer(x)

        return x


class UnifiedSkillNet(nn.Module):

    def __init__(self, input_dim_list, embedding_dim_list, instance_norm_flags,
                 middle_dim_list, middle_dim_other, num_targets,
                 num_layers_attend, num_layers_assess, heavy_assess_head):

        super(UnifiedSkillNet, self).__init__()

        self.num_feature_types = len(input_dim_list)

        self.prepare_module = PreparationModule(input_dim_list,
                                                embedding_dim_list, instance_norm_flags)

        prepared_dim_list = self.prepare_module.get_prepared_dim()

        self.assess_modules_base = nn.ModuleList([
            SingleStageTCN(
                num_layers=num_layers_assess,
                input_dim=prepared_dim_list[i],
                middle_dim=middle_dim_list[i],
                output_dim=middle_dim_list[i],
                dropout=0.5) if middle_dim_list[i] else None
            for i in range(self.num_feature_types)
        ])

        self.assess_modules_head = []
        for i in range(self.num_feature_types):
            if middle_dim_list[i]:
                if heavy_assess_head[i]:
                    self.assess_modules_head.append(
                        ChannelMLPModule(
                            input_dim=middle_dim_list[i],
                            middle_dim=middle_dim_list[i],
                            output_dim=num_targets,
                            num_layers=2)
                    )
                else:
                    self.assess_modules_head.append(
                        nn.Conv1d(middle_dim_list[i], num_targets, 1)
                    )
            else:
                self.assess_modules_head.append(None)

        self.assess_modules_head = nn.ModuleList(self.assess_modules_head)

        self.attend_dropout = nn.Dropout2d(p=0.25)
        self.attend_modules = nn.ModuleList([
            ChannelMLPModule(
                input_dim=self.prepare_module.get_total_dim(),
                middle_dim=middle_dim_other,
                output_dim=1,
                num_layers=num_layers_attend) if middle_dim_list[i] else None
            for i in range(self.num_feature_types)
        ])

        self.fusion_weights = nn.Parameter(torch.zeros(num_targets, self.num_feature_types))

    def forward(self, x, weighting=False):

        # Input: [(N, F, T)]   Output: (N, S), (N, S, P), (1, S, P)
        # S: num_targets, P: num_feature_types

        scores, attentions = self.get_score_attention(x)
        # [TO BE IMPROVED] Current attention is the same for different targets

        scores = [(scores[i] * attentions[i]).sum(2)
                  for i in range(self.num_feature_types)]
        scores = torch.cat([i.unsqueeze(2) for i in scores], dim=2)  # (N, S, P)

        fusion_weights = self.fusion_weights.unsqueeze(0)  # (1, S, P)

        if weighting:
            fused_score = (scores.detach() * F.softmax(fusion_weights, dim=2)).sum(2)
        else:
            fused_score = scores.detach().mean(2)

        return fused_score, scores, fusion_weights

    def get_codings(self, x):

        x = self.prepare_module(x)

        codings = [self.assess_modules_base[i](x[i])
                   if self.assess_modules_base[i] else x[i]
                   for i in range(self.num_feature_types)]

        return codings

    def get_score_attention(self, x):

        # Input: [(N, F, T)]   Output: [(N, S, T)],  [(N, 1, T)]

        x = self.prepare_module(x)

        codings = [self.assess_modules_base[i](x[i])
                   if self.assess_modules_base[i] else x[i]
                   for i in range(self.num_feature_types)]

        scores = [self.assess_modules_head[i](codings[i])
                  if self.assess_modules_head[i] else codings[i]
                  for i in range(self.num_feature_types)]

        x_cat = torch.cat(x, dim=1)
        x_cat = self.attend_dropout(x_cat.unsqueeze(3)).squeeze(3)

        attentions = [F.softmax(self.attend_modules[i](x_cat), dim=2)
                      if self.attend_modules[i] else F.softmax(torch.zeros_like(scores[i]), dim=2)
                      for i in range(self.num_feature_types)]

        return scores, attentions


class CodingPredictor(nn.Module):

    def __init__(self, num_feature_types, middle_dim_list, num_layers):
        super(CodingPredictor, self).__init__()

        self.num_feature_types = num_feature_types

        self.predictors = nn.ModuleList([
            ChannelMLPModule(
                input_dim=middle_dim_list[i],
                middle_dim=middle_dim_list[i],
                output_dim=middle_dim_list[i],
                num_layers=num_layers) if middle_dim_list[i] else None
            for i in range(num_feature_types)
        ])

    def forward(self, x):  # Input: [(N, F, T)]   Output: [(N, F, T)]

        futures = [self.predictors[i](x[i])
                   if self.predictors[i] else None
                   for i in range(self.num_feature_types)]

        return futures
