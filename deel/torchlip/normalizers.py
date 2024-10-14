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

DEFAULT_EPS_SPECTRAL = 1e-3
DEFAULT_EPS_BJORCK = 1e-3
DEFAULT_MAXITER_BJORCK = 15
DEFAULT_MAXITER_SPECTRAL = 10
DEFAULT_BETA = 0.5


def bjorck_normalization(
    w: torch.Tensor,
    eps: float = DEFAULT_EPS_BJORCK,
    beta: float = DEFAULT_BETA,
    maxiter: int = DEFAULT_MAXITER_BJORCK,
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
        eps (float): epsilon stopping criterion: norm(wt - wt-1) must be less than eps
        beta (float): beta used in each iteration, must be in the interval ]0, 0.5]
        maxiter (int): maximum number of iterations for the algorithm

    Returns:
        The weights :math:`\overline{W}` after Bjorck normalization.
    """

    def cond(w, old_w):
        return torch.linalg.norm(w - old_w) >= eps

    # define the loop body
    def body_cols(w):
        return torch.mm(w, torch.mm(w.t(), w))

    def body_rows(w):
        return torch.mm(torch.mm(w, w.t()), w)

    def body(w, fct):
        w = (1.0 + beta) * w - beta * fct(w)
        return w

    shape = w.shape
    cout = w.size(0)
    w_mat = w.reshape(cout, -1)

    if w_mat.shape[0] > w_mat.shape[1]:
        body_fct = body_cols
    else:
        body_fct = body_rows

    done = False
    iter = maxiter

    while not done:
        old_w = w_mat
        w_mat = body(w_mat, body_fct)
        iter -= 1
        done = (not cond(w_mat, old_w)) or (iter <= 0)

    w = w_mat.reshape(shape)
    return w


def _power_iteration(
    linear_operator,
    adjoint_operator,
    u: torch.Tensor,
    eps=DEFAULT_EPS_SPECTRAL,
    maxiter=DEFAULT_MAXITER_SPECTRAL,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Internal function that performs the power iteration algorithm.

    Args:
        linear_operator (Callable): a callable object that maps a linear operation.
        adjoint_operator (Callable): a callable object that maps the adjoint of the
            linear operator.
        u (tf.Tensor): initialization of the singular vector.
        eps (float, optional): stopping criterion of the algorithm, when
            norm(u[t] - u[t-1]) is less than eps. Defaults to DEFAULT_EPS_SPECTRAL.
        maxiter (int, optional): maximum number of iterations for the algorithm.
            Defaults to DEFAULT_MAXITER_SPECTRAL.

    Returns:
         A Tensor containing the maximum singular vector

    """

    # Loop stopping condition
    def cond(u, old_u):
        return torch.linalg.norm(u - old_u) >= eps

    # Loop body
    def body(u):
        v = linear_operator(u)
        u = adjoint_operator(v)

        u = F.normalize(u, dim=-1)

        return u

    done = False
    iter = maxiter

    while not done:
        old_u = u
        u = body(u)
        iter -= 1
        done = (not cond(u, old_u)) or (iter <= 0)

    return u


def spectral_normalization(
    kernel: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    eps: float = DEFAULT_EPS_SPECTRAL,
    maxiter: int = DEFAULT_MAXITER_SPECTRAL,
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
        eps (float, optional): stopping criterion of the algorithm, when
            norm(u[t] - u[t-1]) is less than eps. Defaults to DEFAULT_EPS_SPECTRAL.
        maxiter (int, optional): maximum number of iterations for the algorithm.
            Defaults to DEFAULT_MAXITER_SPECTRAL.

    Returns:
        The normalized kernel :math:`\overline{W}`, the right singular vector
        corresponding to the largest singular value, and the largest singular value
        before normalization.
    """

    def linear_op(u):
        return u @ kernel.t()

    def adjoint_op(v):
        return v @ kernel

    # Flatten the Tensor
    W_flat = kernel.flatten(start_dim=1)

    if u is None:
        maxiter *= 2  # if u was not double number of iterations for the first run
        u = torch.ones(tuple([1, W_flat.shape[-1]]), device=kernel.device)
        torch.nn.init.normal_(u)

    # do power iteration
    u = _power_iteration(linear_op, adjoint_op, u, eps, maxiter)

    # Compute the largest singular value and the normalized kernel.
    # We assume that in the worst case we converged to sigma + eps (as u and v are
    # normalized after each iteration)
    # In order to be sure that operator norm of normalized kernel is strictly less than
    # one we use sigma + eps, which ensures stability of Björck algorithm even when
    # beta=0.5
    sigma = torch.linalg.norm(linear_op(u))
    normalized_kernel = kernel.div(sigma + eps)
    return normalized_kernel, u, sigma
