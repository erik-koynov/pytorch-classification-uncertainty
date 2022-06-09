import torch
import torch.nn.functional as F
from evidential_tools.helpers import get_device


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)

def dirichlet_kl_divergence(first: torch.Tensor,
                            second: torch.Tensor,
                            strength_1: torch.Tensor = None,
                            strength_2: torch.Tensor = None,
                            reduce_mean=True):
    """
    Compute the KL divergence between two dirichlet distributed rv
    :param first / second : parameters of the distributions : B x C (x H x W)
    :param strength_1 / strength_2: strength of the two distributions - i.e. sums of their paramters
                                    : B x 1 (x H x W)
    :return:
    """
    if strength_1 is None:
        strength_1 = first.sum(dim=1, keepdims=True)
    if strength_2 is None:
        strength_2 = second.sum(dim=1, keepdims=True)

    logBeta_first = torch.lgamma(first).sum(dim=1, keepdims=True) - torch.lgamma(strength_1)
    logBeta_second = torch.lgamma(second).sum(dim=1, keepdims=True) - torch.lgamma(strength_2)

    digamma_term = torch.digamma(first) - torch.digamma(strength_1)

    log_dirichlet_rv = ((first-second)*digamma_term).sum(dim=1, keepdims = True)

    kl_divergence = logBeta_second - logBeta_first + log_dirichlet_rv

    if reduce_mean:
        return kl_divergence.mean()
    else:
        return kl_divergence





def loglikelihood_loss(alpha: torch.Tensor,
                       y: torch.Tensor,
                       strength = None,
                       reduce_mean=True):
    """
    Compute the likelihood loss. Works for dense predictions also.
    :param y: B x C (x H x W)
    :param alpha: B x C (x H x W)
    :return:
    """
    if strength is None:
        strength = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / strength)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (strength - alpha) / (strength * strength * (strength + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    if reduce_mean:
        return loglikelihood.mean()
    else:
        return loglikelihood


def mse_loss(y, alpha, epoch_num, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(alpha, y)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * dirichlet_kl_divergence(kl_alpha, torch.ones_like(kl_alpha))
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * dirichlet_kl_divergence(kl_alpha, torch.ones_like(kl_alpha))
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss
