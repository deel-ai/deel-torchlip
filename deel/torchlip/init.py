# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
"""

import torch

from .normalizers import DEFAULT_NITER_SPECTRAL_INIT, DEFAULT_NITER_BJORCK, DEFAULT_BETA
from .normalizers import spectral_normalization, bjorck_normalization


def spectral_(
    tensor: torch.Tensor, n_power_iterations: int = DEFAULT_NITER_SPECTRAL_INIT
):
    r"""
    Apply spectral normalization on the given tensor in-place.

    See also :py:func:`spectral_normalization`.

    .. warning::
        This function is provided for completeness but we recommend using
        :py:func:`torch.nn.init.orthogonal_` instead to obtain a proper (semi)
        orthogonal matrix.

    Args:
        tensor: A 2-dimensional ``torch.Tensor``.
        n_power_iterations: Number of iterations to perform.
    """
    with torch.no_grad():
        tensor.copy_(spectral_normalization(tensor, None, n_power_iterations)[0])


def bjorck_(
    tensor: torch.Tensor,
    n_iterations: int = DEFAULT_NITER_BJORCK,
    beta: float = DEFAULT_BETA,
):
    r"""
    Apply Bjorck normalization on the given tensor in-place.

    See also :py:func:`bjorck_normalization`.

    .. warning::
        This function is provided for completeness but we recommend using
        :py:func:`torch.nn.init.orthogonal_` instead to obtain a proper (semi)
        orthogonal matrix.

    Args:
        tensor: A 2-dimensional ``torch.Tensor``.
        n_iterations: Number of iterations to perform.
        beta: Value to use for the :math:`\beta` parameter.
    """
    with torch.no_grad():
        tensor.copy_(bjorck_normalization(tensor, n_iterations, beta))
