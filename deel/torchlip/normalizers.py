# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains computation functions for Bjorck and spectral
normalization. This is most for internal use.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

DEFAULT_NITER_BJORCK = 15
DEFAULT_NITER_SPECTRAL = 3
DEFAULT_NITER_SPECTRAL_INIT = 10
DEFAULT_BETA = 0.5


def bjorck_normalization(
    w: torch.Tensor, niter: int = DEFAULT_NITER_BJORCK
) -> torch.Tensor:
    """
    Apply Bjorck normalization on w.

    Args:
        w: Weights to normalize. For the normalization to work properly, the greatest
            eigen value of w must be approximately 1.
        niter: Number of iterations.

    Returns:
        The weights after Bjorck normalization.

    """
    if niter == 0:
        return w
    shape = w.shape
    height = w.size(0)
    w_mat = w.reshape(height, -1)
    for i in range(niter):
        w_mat = (1.0 + DEFAULT_BETA) * w_mat - DEFAULT_BETA * torch.mm(
            w_mat, torch.mm(w_mat.t(), w_mat)
        )
    w = w_mat.reshape(shape)
    return w


def _power_iteration(
    w: torch.Tensor, u: torch.Tensor, niter: int = DEFAULT_NITER_SPECTRAL
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Internal function that performs the power iteration algorithm.

    Args:
        w: Weights matrix that we want to find eigen vector.
        u: Initialization of the eigen vector.
        niter: Number of iteration, must be greater than 0.

    Returns:
         A tuple (u, v) containing the largest eigenvalues.

    """
    for i in range(niter):
        v = F.normalize(torch.mm(u, w.t()), p=2, dim=1)
        u = F.normalize(torch.mm(v, w), p=2, dim=1)
    return u, v


def spectral_normalization(
    kernel: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    niter: int = DEFAULT_NITER_SPECTRAL,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the kernel to have it's max eigenvalue == 1.

    Args:
        kernel: The kernel to normalize.
        u: Initialization for the maximum eigen vector.
        niter: Number of iteration. If u is not specified, we perform
            twice as much iterations.

    Returns:
        The normalized kernel W_bar, the maximum eigen vector, and the
        largest eigen value.
    """

    # Flatten the Tensor
    W_flat = kernel.flatten(start_dim=1)

    if u is None:
        niter *= 2  # if u was not double number of iterations for the first run
        u = torch.ones(tuple([1, W_flat.shape[-1]]), device=kernel.device)
        torch.nn.init.normal_(u)

    # do power iteration
    u, v = _power_iteration(W_flat, u, niter)

    # Calculate Sigma
    sigma = v.mm(W_flat).mm(u.t())

    # Normalize W_bar
    W_bar = kernel.div(sigma)

    return W_bar, u, sigma
