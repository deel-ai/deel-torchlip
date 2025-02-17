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
""" """
import warnings
import torch

from .normalizers import bjorck_normalization
from .normalizers import DEFAULT_BETA
from .normalizers import DEFAULT_EPS_SPECTRAL
from .normalizers import DEFAULT_MAXITER_SPECTRAL
from .normalizers import DEFAULT_EPS_BJORCK
from .normalizers import DEFAULT_MAXITER_BJORCK
from .normalizers import spectral_normalization


def spectral_(
    tensor: torch.Tensor,
    eps_spectral: float = DEFAULT_EPS_SPECTRAL,
    maxiter_spectral: int = DEFAULT_MAXITER_SPECTRAL,
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
        eps_spectral (float): stopping criterion of iterative power method
        maxiter_spectral (int): maximum number of iterations for the power iteration
    """
    warnings.warn(
        "spectral_ initialization is deprecated, use torch.nn.init.orthogonal_ instead"
    )
    with torch.no_grad():
        tensor.copy_(
            spectral_normalization(
                tensor, None, eps=eps_spectral, maxiter=maxiter_spectral
            )[0]
        )


def bjorck_(
    tensor: torch.Tensor,
    eps_spectral: float = DEFAULT_EPS_SPECTRAL,
    maxiter_spectral: int = DEFAULT_MAXITER_SPECTRAL,
    eps_bjorck=DEFAULT_EPS_BJORCK,
    maxiter_bjorck: int = DEFAULT_MAXITER_BJORCK,
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
        eps_spectral (float): stopping criterion of iterative power method
        maxiter_spectral (int): maximum number of iterations for the power iteration
        eps_bjorck (float): stopping criterion in bjorck algorithm
        maxiter_bjorck (int): maximum number of iterations for bjorck algorithm
        beta: Value to use for the :math:`\beta` parameter.
    """
    warnings.warn(
        "bjorck_ initialization is deprecated, use torch.nn.init.orthogonal_ instead"
    )
    with torch.no_grad():
        spectral_tensor = spectral_normalization(
            tensor, None, eps=eps_spectral, maxiter=maxiter_spectral
        )[0]
        tensor.copy_(
            bjorck_normalization(
                spectral_tensor, eps=eps_bjorck, beta=beta, maxiter=maxiter_bjorck
            )
        )
