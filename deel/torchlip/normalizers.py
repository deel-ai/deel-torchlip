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
This module contains computation functions for Bjorck and spectral
normalization. This is most for internal use.
"""
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F

DEFAULT_NITER_BJORCK = 15
DEFAULT_NITER_SPECTRAL = 3
DEFAULT_NITER_SPECTRAL_INIT = 10
DEFAULT_BETA = 0.5


def bjorck_normalization(
    w: torch.Tensor, niter: int = DEFAULT_NITER_BJORCK, beta: float = DEFAULT_BETA
) -> torch.Tensor:
    r"""
    Apply Bjorck normalization on the kernel as per

    .. math::
        \begin{array}{l}
            W_0 = W \\
            W_{n + 1} = (1 + \beta) W_{n} - \beta W_n W_n^T W_n \\
            \overline{W} = W_{N}
        \end{array}

    where :math:`W` is the kernel of shape :math:`(C, *)`, with :math:`C` the number
    of channels.

    Args:
        w: Weights to normalize. For the normalization to work properly, the greatest
            eigen value of ``w`` must be approximately 1.
        niter: Number of iterations.
        beta: Value of :math:`\beta` to use.

    Returns:
        The weights :math:`\overline{W}` after Bjorck normalization.
    """
    if niter == 0:
        return w
    shape = w.shape
    cout = w.size(0)
    w_mat = w.reshape(cout, -1)
    for i in range(niter):
        w_mat = (1.0 + beta) * w_mat - beta * torch.mm(
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
        u: Initial singular vector.
        niter: Number of iteration, must be greater than 0.

    Returns:
         A tuple (u, v) containing the last singular vectors.

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
    r"""
    Apply spectral normalization on the given kernel :math:`W` to set its greatest
    eigen value to approximately 1.

    .. note::
        This function is provided for completeness since ``torchlip`` layers use
        the :py:func:`torch.nn.utils.spectral_norm` hook instead.

    Args:
        kernel: The kernel to normalize.
        u: Initialization for the initial singular vector. If ``None``, it will
            be randomly initialized from a normal distribution and twice
            as much iterations will be performed.
        niter: Number of iteration. If u is not specified, we perform
            twice as much iterations.

    Returns:
        The normalized kernel :math:`\overline{W}`, the right singular vector
        corresponding to the largest singular value, and the largest singular value
        before normalization.
    """

    # Flatten the Tensor
    W_flat = kernel.flatten(start_dim=1)

    if u is None:
        niter *= 2  # if u was not double number of iterations for the first run
        u = torch.ones(tuple([1, W_flat.shape[-1]]), device=kernel.device)
        torch.nn.init.normal_(u)

    # do power iteration
    u, v = _power_iteration(W_flat, u, niter)

    # Calculate sigma (largest singular value).
    sigma = v.mm(W_flat).mm(u.t())

    # Normalize W_bar
    W_bar = kernel.div(sigma)

    return W_bar, u, sigma
