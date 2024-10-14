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
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from ..normalizers import bjorck_normalization
from ..normalizers import DEFAULT_NITER_BJORCK


class _BjorckNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, n_iterations: int) -> None:
        super().__init__()
        self.n_iterations = n_iterations
        self.register_buffer("_w_bjorck", weight.data)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_bjorck = bjorck_normalization(weight, self.n_iterations)
            self._w_bjorck = w_bjorck.data
        else:
            w_bjorck = self._w_bjorck
        return w_bjorck


def bjorck_norm(
    module: nn.Module, name: str = "weight", n_iterations: int = DEFAULT_NITER_BJORCK
) -> nn.Module:
    r"""
    Applies Bjorck normalization to a parameter in the given module.

    Bjorck normalization ensures that all eigen values of a vectors remain close or
    equal to one during training. If the dimension of the weight tensor is greater than
    2, it is reshaped to 2D for iteration.
    This is implemented via a Bjorck normalization parametrization.

    .. note::
        It is recommended to use :py:func:`torch.nn.utils.parameterize.spectral_norm`
        before this hook to greatly reduce the number of iterations required.

    See `Sorting out Lipschitz function approximation
    <https://arxiv.org/abs/1811.05381>`_.

    Args:
        module: Containing module.
        name: Name of weight parameter.
        n_iterations: Number of iterations for the normalization.

    Returns:
        The original module with the Bjorck normalization hook.

    Example:

        >>> m = bjorck_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)

    See Also:
        :py:func:`deel.torchlip.normalizers.bjorck_normalization`
    """
    weight = getattr(module, name, None)
    parametrize.register_parametrization(
        module, name, _BjorckNorm(weight, n_iterations)
    )
    return module


def remove_bjorck_norm(module: nn.Module, name: str = "weight") -> nn.Module:
    r"""
    Removes the Bjorck normalization reparameterization from a module.

    Args:
        module: Containing module.
        name: Name of weight parameter.

    Example:
        >>> m = bjorck_norm(nn.Linear(20, 40))
        >>> remove_bjorck_norm(m)
    """
    for key, m in module.parametrizations[name]._modules.items():
        if isinstance(m, _BjorckNorm):
            if len(module.parametrizations["weight"]) == 1:
                parametrize.remove_parametrizations(module, name)
            else:
                del module.parametrizations[name]._modules[key]
