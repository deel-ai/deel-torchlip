# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Contains utility functions.
"""
from typing import Optional

import torch

from .bjorck_norm import bjorck_norm
from .bjorck_norm import remove_bjorck_norm
from .frobenius_norm import frobenius_norm
from .frobenius_norm import remove_frobenius_norm
from .lconv_norm import lconv_norm
from .lconv_norm import remove_lconv_norm
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

    return float(lip_cst.item())


__all__ = [
    "bjorck_norm",
    "remove_bjorck_norm",
    "frobenius_norm",
    "remove_frobenius_norm",
    "lconv_norm",
    "remove_lconv_norm",
    "sqrt_with_gradeps",
    "evaluate_lip_const",
]
