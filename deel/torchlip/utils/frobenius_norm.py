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


class _FrobeniusNorm(nn.Module):
    def __init__(self, disjoint_neurons: bool) -> None:
        super().__init__()
        self.dim_norm = 1 if disjoint_neurons else None

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight / torch.norm(weight, dim=self.dim_norm, keepdim=True)


def frobenius_norm(
    module: nn.Module, name: str = "weight", disjoint_neurons: bool = True
) -> nn.Module:
    r"""
    Applies Frobenius normalization to a parameter in the given module.

    .. math::
         \mathbf{W} = \dfrac{\mathbf{W}}{\Vert{}\mathbf{W}\Vert{}}

    This is implemented via a hook that applies Frobenius normalization before every
    ``forward()`` call.

    Args:
        module: Containing module.
        name: Name of weight parameter.
        disjoint_neurons: Normalize, independently per neuron or not, the matrix weight.

    Returns:
        The original module with the Frobenius normalization hook.

    Example::

        >>> m = frobenius_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)

    """
    parametrize.register_parametrization(module, name, _FrobeniusNorm(disjoint_neurons))
    return module


def remove_frobenius_norm(module: nn.Module, name: str = "weight") -> nn.Module:
    r"""
    Removes the Frobenius normalization reparameterization from a module.

    Args:
        module: Containing module.
        name: Name of weight parameter.

    Example:

        >>> m = frobenius_norm(nn.Linear(20, 40))
        >>> remove_frobenius_norm(m)
    """
    for key, m in module.parametrizations[name]._modules.items():
        if isinstance(m, _FrobeniusNorm):
            if len(module.parametrizations["weight"]) == 1:
                parametrize.remove_parametrizations(module, name)
            else:
                del module.parametrizations[name]._modules[key]
