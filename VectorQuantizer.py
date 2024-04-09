import torch_scatter
from torch_geometric.utils import get_embeddings
import torch
from torch import nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25,
                 delta: float = 1):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.delta = delta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents_mean, latents_std):
        latents_shape = latents_mean.shape
        flat_latents = latents_mean.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents_mean and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents_mean.device
        encoding_one_hot_mean = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot_mean.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents_mean
        quantized_latents_mean = torch.matmul(encoding_one_hot_mean, self.embedding.weight)  # [BHW, D]
        quantized_latents_mean = quantized_latents_mean.view(latents_shape)  # [B x H x W x D]



        latents_shape = latents_std.shape
        flat_latents = latents_std.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents_std and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents_std.device
        encoding_one_hot_std = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot_std.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents_std
        quantized_latents_std = torch.matmul(encoding_one_hot_std, self.embedding.weight)  # [BHW, D]
        quantized_latents_std = quantized_latents_std.view(latents_shape)  # [B x H x W x D]




        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents_mean.detach(), latents_mean)
        commitment_loss += F.mse_loss(quantized_latents_std.detach(), latents_std)
        # 改成两个高斯分布的loss
        # embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        embedding_loss = cal_kl_loss(latents_mean, latents_std, quantized_latents_mean.detach(), quantized_latents_std.detach())

        vq_loss = commitment_loss * self.beta + self.delta * embedding_loss



        # Add the residue back to the latents
        quantized_latents_mean = latents_mean + (quantized_latents_mean - latents_mean).detach()
        avg_probs_mean = torch.mean(encoding_one_hot_mean, dim=0)
        perplexity_mean = torch.exp(-torch.sum(avg_probs_mean * torch.log(avg_probs_mean + 1e-10)))
        # print('perplexity_mean: ',perplexity_mean)

        avg_probs_std = torch.mean(encoding_one_hot_std, dim=0)
        # print('avg_probs:', avg_probs_std)
        perplexity_std = torch.exp(-torch.sum(avg_probs_std * torch.log(avg_probs_std + 1e-10)))
        # print('perplexity_mean: ',perplexity_std)

        return quantized_latents_mean, quantized_latents_std, vq_loss, perplexity_mean, perplexity_std

    def sample(self, latents_mean, latents_std):
        # Convert to one-hot encodings
        device = latents_mean.device    
        latents_shape = latents_mean.shape
        flat_latents = latents_mean.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents_mean and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]
               
        # encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
        # 从距离矩阵中随机选择一个索引
        encoding_inds = torch.randint(dist.size(1), (dist.size(0), 1), dtype=torch.long, device=device)  # [BHW, 1]

        
        encoding_one_hot_mean = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot_mean.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents_mean
        quantized_latents_mean = torch.matmul(encoding_one_hot_mean, self.embedding.weight)  # [BHW, D]
        quantized_latents_mean = quantized_latents_mean.view(latents_shape)  # [B x H x W x D]

        latents_shape = latents_std.shape
        flat_latents = latents_std.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents_std and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
        # 从距离矩阵中随机选择一个索引
        encoding_inds = torch.randint(dist.size(1), (dist.size(0), 1), dtype=torch.long, device=device)  # [BHW, 1]
        # replace 随机one hoc编码

        # Convert to one-hot encodings
        device = latents_std.device
        encoding_one_hot_std = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot_std.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents_std
        quantized_latents_std = torch.matmul(encoding_one_hot_std, self.embedding.weight)  # [BHW, D]
        quantized_latents_std = quantized_latents_std.view(latents_shape)  # [B x H x W x D]

        # Add the residue back to the latents
        quantized_latents_mean = latents_mean + (quantized_latents_mean - latents_mean).detach()
        avg_probs_mean = torch.mean(encoding_one_hot_mean, dim=0)
        perplexity_mean = torch.exp(-torch.sum(avg_probs_mean * torch.log(avg_probs_mean + 1e-10)))
        # print('perplexity_mean: ',perplexity_mean)

        avg_probs_std = torch.mean(encoding_one_hot_std, dim=0)
        # print('avg_probs:', avg_probs_std)
        perplexity_std = torch.exp(-torch.sum(avg_probs_std * torch.log(avg_probs_std + 1e-10)))
        # print('perplexity_mean: ',perplexity_std)

        return quantized_latents_mean, quantized_latents_std

def cal_kl_loss(mu_poster=None, sigma_poster=None, mu_prior=None, sigma_prior=None):
    eps = 10 ** -8
    sigma_poster = sigma_poster ** 2
    sigma_prior = sigma_prior ** 2
    sigma_poster_matrix_det = torch.prod(sigma_poster, dim=1)
    sigma_prior_matrix_det = torch.prod(sigma_prior, dim=1)

    sigma_prior_matrix_inv = 1.0 / sigma_prior
    delta_u = (mu_prior - mu_poster)
    term1 = torch.sum(sigma_poster / sigma_prior, dim=1)
    term2 = torch.sum(delta_u * sigma_prior_matrix_inv * delta_u, 1)
    term3 = - mu_poster.shape[-1]
    term4 = torch.log(sigma_prior_matrix_det + eps) - torch.log(
        sigma_poster_matrix_det + eps)
    kl_loss = 0.5 * (term1 + term2 + term3 + term4)
    kl_loss = torch.clamp(kl_loss, 0, 10)

    return torch.mean(kl_loss)