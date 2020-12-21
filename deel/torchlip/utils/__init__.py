# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Contains utility functions.
"""

from typing import Optional

import torch

from .bjorck_norm import bjorck_norm, remove_bjorck_norm  # noqa: F401
from .frobenius_norm import frobenius_norm, remove_frobenius_norm  # noqa: F401
from .lconv_norm import lconv_norm, remove_lconv_norm  # noqa: F401

from .sqrt_eps import sqrt_with_gradeps  # noqa: F401

DEFAULT_NITER_BJORCK = 15
DEFAULT_NITER_SPECTRAL = 3
DEFAULT_NITER_SPECTRAL_INIT = 10


def evaluate_lip_const(
    model: torch.nn.Module,
    x: torch.Tensor,
    eps: float = 1e-4,
    seed: Optional[int] = None,
) -> float:
    """
    Evaluate the Lipschitz constant of a model, with the naive method.
    Please note that the estimation of the lipschitz constant is done locally around
    input sample. This may not correctly estimate the behavior in the whole domain.

    Args:
        model: Torch model used to make predictions.
        x: Inputs used to compute the lipschitz constant.
        eps: Magnitude of noise to add to input in order to compute the constant.
        seed: Seed used when generating the noise. If None, a random seed will be
            used.

    Returns:
        The empirically evaluated lipschitz constant. The computation might also be
        inaccurate in high dimensional space.

    """
    y_pred = model(x.float())

    with torch.random.fork_rng():
        if seed is None:
            torch.random.seed()
        else:
            torch.random.manual_seed(seed)
        x_var = x + torch.distributions.Uniform(low=eps * 0.25, high=eps).sample(
            x.shape
        )

    y_pred_var = model(x_var.float())

    dx = x - x_var
    dfx = y_pred - y_pred_var
    ndx = torch.sum(torch.square(dx), dim=tuple(range(1, len(x.shape))))
    ndfx = torch.sum(torch.square(dfx.data), dim=tuple(range(1, len(y_pred.shape))))
    lip_cst = torch.sqrt(torch.max(ndfx / ndx))

    return lip_cst.item()
