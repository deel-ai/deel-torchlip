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

if torch.__version__.startswith("1."):
    import functorch as tfc
else:
    import torch.func as tfc

from .bjorck_norm import bjorck_norm
from .bjorck_norm import remove_bjorck_norm
from .frobenius_norm import frobenius_norm
from .frobenius_norm import remove_frobenius_norm
from .lconv_norm import lconv_norm
from .lconv_norm import remove_lconv_norm
from .sqrt_eps import sqrt_with_gradeps  # noqa: F401


def evaluate_lip_const(
    model: torch.nn.Module,
    x: torch.Tensor,
    eps: float = 1e-4,
    seed: Optional[int] = None,
) -> float:
    """
    Evaluate the Lipschitz constant of a model, using the Jacobian of the model.
    Please note that the estimation of the lipschitz constant is done locally around
    input samples. This may not correctly estimate the behaviour in the whole domain.

    Args:
        model: built keras model used to make predictions
        x: inputs used to compute the lipschitz constant

    Returns:
        float: the empirically evaluated Lipschitz constant. The computation might also
            be inaccurate in high dimensional space.

    """

    def model_func(x):
        y = model(torch.unsqueeze(x, dim=0))  # Forward pass
        return y

    x_src = x.clone().detach().requires_grad_(True)

    # Compute the Jacobian using jacrev
    batch_jacobian = tfc.vmap(tfc.jacrev(model_func))(x_src)

    # Reshape the Jacobian to match the desired shape
    batch_size = x.shape[0]
    xdim = torch.prod(torch.tensor(x.shape[1:])).item()
    batch_jacobian = batch_jacobian.view(batch_size, -1, xdim)

    # Compute singular values and check Lipschitz property
    lip_cst = torch.linalg.norm(batch_jacobian, ord=2, dim=(-2, -1))
    return float(torch.max(lip_cst).item())


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
