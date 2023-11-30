import torch.nn
from torch.nn import functional as F
from hyperparams import device

def get_vae_loss(x, x_recon, mu_xh, logvar_xh, beta=1.0):
    recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction="none").sum(
        dim=[1, 2, 3]
    )
    prior_loss = -0.5 * (1 + logvar_xh - mu_xh.pow(2) - logvar_xh.exp()).sum(dim=1)
    loss = recon_loss + beta * prior_loss
    return loss.mean(), recon_loss.mean(), prior_loss.mean()


def get_yh_pred_loss(yh_prime, yh):
    mse = F.mse_loss(yh_prime, yh, reduction="none")
    var = yh_prime.var(dim=0, keepdim=True).clamp_min(1e-6)
    loss = (mse / var).sum(dim=1)
    return loss.mean()


def get_auxiliary_classifier_loss(pred, label):
    return F.cross_entropy(pred, label)

def get_sparsity_loss(f_xy, lambda2=1.0, d=1):
    # check for identity
    if f_xy is None:
        return torch.tensor(0.0).to(device)
    # L1 loss for model.f_xy.parameters()
    l1_loss = torch.tensor(0.0).to(device)
    for param in f_xy.parameters():
        # dim=1 is row wise, dim=0 is column wise, currently only a single linear layer
        l1_loss+= torch.mean(torch.norm(param, p=1, dim=d)) 
    return lambda2*l1_loss