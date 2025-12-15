import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from torch.nn import Parameter


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class DyHG(nn.Module):
    def __init__(self, in_dim=1024, emb_dim=256, n_classes=8, dropout=0.25, hyper_num=20, num_layers=1, tau=0.05,
                 device='cuda:0'):
        super().__init__()

        self.message_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(emb_dim, n_classes)
        self.num_layers = num_layers
        self.device = device
        self.hyper_num = hyper_num
        self.act = nn.LeakyReLU()
        self.norm = nn.LayerNorm(emb_dim)
        self._fc1 = nn.Sequential(nn.Linear(in_dim, emb_dim), nn.ReLU(), nn.Dropout(dropout))
        self.tau = tau
        self.n_classes = n_classes
        size = [emb_dim, emb_dim, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]

        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)

        self.construct_hyper = nn.Sequential(
            nn.Linear(emb_dim, self.hyper_num),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hyper_num)
        )

    def gumbel_softmax_sample(self, logits, temperature=1.0, hard=False):
        return F.gumbel_softmax(logits, tau=temperature, hard=hard)

    def forward(self, x):
        try:
            x = x["feature"]
        except:
            x = x

        x = self._fc1(x)  # [B, N, C]
        features = [x]
        B, N, C = x.shape

        patchs_hyper = self.construct_hyper(x)
        patchs_hyper = self.gumbel_softmax_sample(patchs_hyper, temperature=self.tau, hard=False)  # [N, H]
        hyperedge_feature = self.act(patchs_hyper.permute(0, 2, 1) @ x)
        x = self.act(patchs_hyper @ hyperedge_feature)
        x = self.message_dropout(x)
        x = F.normalize(x, p=2, dim=1)
        features.append(x)

        features = torch.stack(features, 1)
        features = torch.sum(features, dim=1).squeeze(1)
        features = features / (self.num_layers + 1)
        features = self.message_dropout(features)
        features = self.norm(features)

        A, h = self.attention_net(features.squeeze(0))
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        M = torch.mm(A, h)  # n_classes, dim

        logits = self.classifiers(M)

        return logits


# demo
if __name__ == "__main__":
    device = 'cuda:0'
    data = torch.randn((1, 200000, 1024)).to(device)
    label = torch.ones((1, 1)).to(device)
    model = DyHG(in_dim=1024, emb_dim=512, n_classes=2, dropout=0.1, hyper_num=16, device=device).to(device)
    output = model(data)
    print(output.shape)

