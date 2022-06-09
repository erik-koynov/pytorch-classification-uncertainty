import torch
import torch.nn.functional as F
import torch.nn as nn
from evidential_tools.losses import loglikelihood_loss, dirichlet_kl_divergence

def alpha_hat_for_kl(alphas: torch.Tensor, targets: torch.Tensor):
    """
    Transform the alphas in order to compute the KL divergence between the distribution described
    by the evidence for the false classes and the uniform distribution
    :param alphas: B x C ( x H x W)
    :param targets: B x C ( x H x W) - should be one-hot-(like) encoded
    :return: B x C ( x H x W)
    """
    return (alphas - 1) * (1 - targets) + 1


def dense_output_activation_head(logits: torch.Tensor):
    """
    Compute the dense activations and dense strength.
    :param logits: output layer preactivations : (B x C x H x W)
    :return: (B x C x H x W) , (B x 1 x H x W)
    """
    evidence = F.relu(logits)
    strength = (evidence+1).sum(dim=1, keepdims=True)
    return evidence+1, strength


def one_hot_encode(targets: torch.Tensor,
                   num_classes=2,
                   soft_binarize=False,
                   hard_binarize=False):
    """
    Return one-hot-encoding or one-hot-like encoding (if the soft labels have been chosen).
    :param targets: B x H x W
    :param num_classes: number of classes : to be set because not all targets contain multiple classes
    :param soft_binarize: whether to prepare the targets as binary soft labels : by convension the
                          0 class will be encoded in the 0th position thus 1-targets encodes the 0 class
    :param hard_binarize: whether to threshold binarization on the targets on the criterion whether they are
                          greater than 0
    :return: B x C x H x W
    """
    if soft_binarize:
        return torch.cat(((1 - targets)[:, None, :], targets[:, None, :]), dim=1)

    if hard_binarize:
        targets = (targets > 0).type(torch.int64)

    return F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)




class LogLikelihoodEvidenceLoss(nn.Module):
    def __init__(self,                    
                 num_classes=2,
                 annealing_factor=10,
                 soft_binarize=False,
                 hard_binarize=False):
        super().__init__()
        self.num_classes = num_classes
        self.soft_binarize = soft_binarize
        self.hard_binarize = hard_binarize
        self.annealing_factor = annealing_factor

    def forward(self, alphas: torch.Tensor, targets: torch.Tensor, epoch: int,  reduce_mean=True):
        """
        Wrapper around loglikelihood loss function with one-hot-encoding of the targets
        :param alphas: B x C (x H x W)
        :param targets: B x H x W
        :return:
        """
        targets = one_hot_encode(targets, self.num_classes, self.soft_binarize, self.hard_binarize)

        annealing_coefficient = min(1.0, float(epoch)/self.annealing_factor)

        llloss = loglikelihood_loss(alphas, targets, reduce_mean=reduce_mean)

        kl_divergence = dirichlet_kl_divergence(alpha_hat_for_kl(alphas, targets),
                                                torch.ones_like(alphas),
                                                reduce_mean=reduce_mean)
        return llloss + annealing_coefficient * kl_divergence

