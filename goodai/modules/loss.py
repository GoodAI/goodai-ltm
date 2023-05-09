import torch
from torch import nn


class EmbCrossProbLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.dist_param = nn.Parameter(torch.ones((1, 1)))

    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor):
        # embeddings_a: (batch_size, num_rk, emb_dim,)
        # embeddings_b: (batch_size, num_sk, emb_dim,)
        batch_size = embeddings_a.size(0)
        num_rk = embeddings_a.size(1)
        num_sk = embeddings_b.size(1)
        embeddings_a = embeddings_a.view(batch_size * num_rk, -1)
        embeddings_b = embeddings_b.view(batch_size * num_sk, -1)
        c_dist = torch.cdist(embeddings_a, embeddings_b)
        # c_dist: (batch_size * num_rk, batch_size * num_sk,)
        c_dist = c_dist.view(batch_size, num_rk, batch_size, num_sk)
        c_dist = torch.amin(c_dist, dim=1)
        # c_dist: (batch_size, batch_size, num_sk,)
        c_dist = torch.amin(c_dist, dim=2)
        # c_dist: (batch_size, batch_size,)
        c_prob_logs = -(c_dist * (5.0 * self.dist_param)).pow(2)
        label = torch.arange(0, batch_size, dtype=torch.long, device=embeddings_a.device)
        return (self.criterion(c_prob_logs, label) + self.criterion(c_prob_logs.transpose(0, 1), label)) / 2

    def get_dist_param_scalar(self) -> float:
        return self.dist_param[0, 0].item()

    @staticmethod
    def get_prob(distances: torch.Tensor, dist_param: float):
        # distances: (batch_size, 1,)
        c_prob_logs = -(distances * dist_param).pow(2)
        return torch.exp(c_prob_logs)
