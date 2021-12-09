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
from typing import Any
from typing import TypeVar

import torch

from .hook_norm import HookNorm


class FrobeniusNorm(HookNorm):
    def __init__(self, module: torch.nn.Module, name: str):
        super().__init__(module, name)

    def compute_weight(self, module: torch.nn.Module, inputs: Any) -> torch.Tensor:
        w: torch.Tensor = self.weight(module)
        return w / torch.norm(w)  # type: ignore

    @staticmethod
    def apply(module: torch.nn.Module, name: str) -> "FrobeniusNorm":
        return FrobeniusNorm(module, name)


T_module = TypeVar("T_module", bound=torch.nn.Module)


def frobenius_norm(module: T_module, name: str = "weight") -> T_module:
    r"""
    Applies Frobenius normalization to a parameter in the given module.

    .. math::
         \mathbf{W} = \dfrac{\mathbf{W}}{\Vert{}\mathbf{W}\Vert{}}

    This is implemented via a hook that applies Bjorck normalization before every
    ``forward()`` call.

    Args:
        module: Containing module.
        name: Name of weight parameter.

    Returns:
        The original module with the Frobenius normalization hook.

    Example::

        >>> m = frobenius_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)

    """
    FrobeniusNorm.apply(module, name)
    return module


def remove_frobenius_norm(module: T_module, name: str = "weight") -> T_module:
    r"""
    Removes the Frobenius normalization reparameterization from a module.

    Args:
        module: Containing module.
        name: Name of weight parameter.

    Example:

        >>> m = frobenius_norm(nn.Linear(20, 40))
        >>> remove_frobenius_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, FrobeniusNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("frobenius_norm of '{}' not found in {}".format(name, module))
