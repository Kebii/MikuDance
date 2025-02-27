import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class MANModule(nn.Module):
    def __init__(self, norm_dim, m_dim, ks=3):
        super().__init__()

        self.norm = nn.InstanceNorm2d(norm_dim, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(m_dim, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_dim, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_dim, kernel_size=ks, padding=pw)

    def forward(self, x, motion_map):

        normalized = self.norm(x)

        motion_map = F.interpolate(motion_map, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(motion_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta

        return out
