import torch
from torch import nn


class ContrastClassifier(nn.Module):
    def __init__(self, scale: float):
        super(ContrastClassifier, self).__init__()
        self.scale = scale
        self.factor = nn.Parameter(torch.ones((1, 1)))
        self.prior_logit = nn.Parameter(torch.zeros((1, 1)))
        self.prior_weight_logit = nn.Parameter(torch.FloatTensor([[-5.0]]))

    def get_min_distance(self, rk: torch.Tensor, sk: torch.Tensor, keepdim=False):
        # rk: (B, num_r_keys, emb_size,)
        # sk: (B, num_s_keys, emb_size,)
        # Returns: (B,) or (B, 1,)
        diff = (sk[:, None, :, :] - rk[:, :, None, :]) * self.scale
        # diff_sq: (B, num_r_keys, num_s_keys, emb_size,)
        mean_diff_sq = torch.mean(diff.pow(2), dim=3)
        # diff_sq: (B, num_r_keys, num_s_keys,)
        s_mins = torch.amin(mean_diff_sq, dim=2)
        # s_mins: (B, num_r_keys,)
        return torch.amin(s_mins, dim=1, keepdim=keepdim)

    def forward(self, r_emb: torch.Tensor, s_emb: torch.Tensor):
        """
        r_emb: Tensor of size (batch_size, 1, emb_size,)
        s_emb: Tensor of size (batch_size, num_keys, emb_size,)
        """
        min_dist_sq = self.get_min_distance(r_emb, s_emb, keepdim=True)
        # min_dist_sq: (batch_size, 1,)
        nominal_p = torch.exp(-min_dist_sq * self.factor.pow(2))
        prior = torch.sigmoid(self.prior_logit)
        prior_weight = torch.sigmoid(self.prior_weight_logit)
        return prior * prior_weight + nominal_p * (1 - prior_weight)
