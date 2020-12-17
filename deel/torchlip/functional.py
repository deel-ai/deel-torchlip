# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

from typing import Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F

# Activations


def max_min(input: torch.Tensor, k_coef_lip: float = 1.0) -> torch.Tensor:
    return torch.cat((F.relu(input), F.relu(-input)), dim=1) * k_coef_lip


def group_sort(
    input: torch.Tensor, group_size: Optional[int] = None, k_coef_lip: float = 1.0
) -> torch.Tensor:
    if group_size is None or group_size > input.shape[1]:
        group_size = input.shape[1]

    if input.shape[1] % group_size != 0:
        raise ValueError("The input size must be a multiple of the group size.")

    fv = input.reshape([-1, group_size])
    if group_size == 2:
        sfv = torch.chunk(fv, 2, 1)
        b = sfv[0]
        c = sfv[1]
        newv = torch.cat((torch.min(b, c), torch.max(b, c)), dim=1)
        newv = newv.reshape(input.shape)
        return newv

    return torch.sort(fv)[0].reshape(input.shape) * k_coef_lip


def group_sort_2(input: torch.Tensor, k_coef_lip: float = 1.0) -> torch.Tensor:
    return group_sort(input, 2, k_coef_lip)


def full_sort(input: torch.Tensor, k_coef_lip: float = 1.0) -> torch.Tensor:
    return group_sort(input, None, k_coef_lip)


def lipschitz_prelu(
    input: torch.Tensor, weight: torch.Tensor, k_coef_lip: float = 1.0
) -> torch.Tensor:
    return F.prelu(input, torch.clamp(weight, -k_coef_lip, +k_coef_lip))


# Losses


def kr_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    true_values: Tuple[int, int] = (0, 1),
) -> torch.Tensor:
    """
    Loss to estimate wasserstein-1 distance using Kantorovich-Rubinstein duality.

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input.
        true_values: tuple containing the two label for each predicted class.

    Returns:
        The Wasserstein-1 loss.
    """

    v0, v1 = true_values
    return torch.mean(input[target == v0]) - torch.mean(input[target == v1])


def neg_kr_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    true_values: Tuple[int, int] = (0, 1),
) -> torch.Tensor:
    """
    Loss to estimate the negative wasserstein-1 distance using Kantorovich-Rubinstein
    duality.

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input.
        true_values: tuple containing the two label for each predicted class.

    Returns:
        The negative Wasserstein-1 loss.
    """
    return -kr_loss(input, target, true_values)


def hinge_margin_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    min_margin: float = 1,
) -> torch.Tensor:
    """
    Compute the hinge margin loss.

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input.
        min_margin: The minimal margin to enforce.

    Returns:
        The hinge margin loss.
    """
    return torch.mean(
        torch.max(torch.zeros_like(input), min_margin - torch.sign(input * target))
    )


def hkr_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    min_margin: float = 1.0,
    true_values: Tuple[int, int] = (0, 1),
) -> torch.Tensor:
    """
    Loss to estimate the wasserstein-1 distance with a hinge regularization using
    Kantorovich-Rubinstein duality.

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input.
        alpha: Regularization factor between the hinge and the KR loss.
        min_margin: Minimal margin for the hinge loss.
        true_values: tuple containing the two label for each predicted class.

    Returns:
        The regularized Wasserstein-1 loss.
    """
    if alpha == np.inf:  # alpha negative hinge only
        return hinge_margin_loss(input, target, min_margin)

    # true value: positive value should be the first to be coherent with the
    # hinge loss (positive y_pred)
    return alpha * hinge_margin_loss(input, target, min_margin) - kr_loss(
        input, target, true_values
    )
