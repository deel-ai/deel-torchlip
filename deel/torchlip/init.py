# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
"""

import torch

from .utils import DEFAULT_NITER_SPECTRAL_INIT, DEFAULT_NITER_BJORCK
from .normalizers import spectral_normalization, bjorck_normalization


def spectral_(
    tensor: torch.Tensor, n_power_iterations: int = DEFAULT_NITER_SPECTRAL_INIT
):
    with torch.no_grad():
        tensor.copy_(spectral_normalization(tensor, None, n_power_iterations)[0])


def bjorck_(tensor: torch.Tensor, n_iterations: int = DEFAULT_NITER_BJORCK):
    with torch.no_grad():
        tensor.copy_(bjorck_normalization(tensor, n_iterations))
