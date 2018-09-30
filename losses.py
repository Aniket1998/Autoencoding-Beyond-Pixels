import torch
import torch.distributions as ds


def reduce(x, reduction=None):
    if reduction == "elementwise_mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    else:
        return x


def kl_loss(mu, logvar, reduction='elementwise_mean'):
    return reduce(-0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar),
                                   dim=1), reduction)


def log_loss(dl_original, dl_recon, reduction='elementwise_mean'):
    dist = ds.Normal(loc=dl_recon, scale=torch.ones_like(dl_recon))
    lp = dist.log_prob(dl_original)
    print(lp.shape)
    return reduce(lp, reduction)


def discriminator_minimax_loss(d_real, d_reconstructed,
                               d_generated, reduction='elementwise_mean'):
    return reduce(torch.log(d_real) + torch.log(1 - d_reconstructed)
                  + torch.log(1 - d_generated), reduction)


def decoder_minimax_loss(d_reconstructed, d_generated,
                         minimax=True, reduction='elementwise_mean'):
    if minimax:
        return reduce(torch.log(1 - d_reconstructed) +
                      torch.log(1 - d_generated), reduction)
    # EXPERIMENTAL
    else:
        return reduce(-(torch.log(d_reconstructed) +
                      torch.log(d_generated)), reduction)


# EXPERIMENTAL
def discriminator_least_squares_loss(d_real, d_reconstructed,
                                     d_generated, a=0.0,
                                     b=1.0, reduction='elementwise_mean'):
    return 0.5 * reduce((d_real - b) ** 2 + (d_generated - a) ** 2 +
                        (d_reconstructed - a) ** 2, reduction)


# EXPERIMENTAL
def decoder_least_squares_loss(d_reconstructed, d_generated,
                               c=1.0, reduction='elementwise_mean'):
    return 0.5 * reduce((d_reconstructed - c) ** 2 +
                        (d_generated - c) ** 2, reduction)
