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
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


def compute_lconv_coef(
    kernel_size: Tuple[int, ...],
    input_shape: Tuple[int, ...],
    strides: Tuple[int, ...] = (1, 1),
) -> float:
    # See https://arxiv.org/abs/2006.06520
    stride = np.prod(strides)
    k1, k2 = kernel_size
    h, w = input_shape[-2:]

    if stride == 1:
        k1_div2 = (k1 - 1) / 2
        k2_div2 = (k2 - 1) / 2
        coefLip = np.sqrt(
            (w * h)
            / ((k1 * h - k1_div2 * (k1_div2 + 1)) * (k2 * w - k2_div2 * (k2_div2 + 1)))
        )
    else:
        sn1 = strides[0]
        sn2 = strides[1]
        coefLip = np.sqrt(1.0 / (np.ceil(k1 / sn1) * np.ceil(k2 / sn2)))

    return coefLip  # type: ignore


class _LConvNorm(nn.Module):
    """Parametrization module for Lipschitz normalization."""

    def __init__(self, lconv_coefficient: float) -> None:
        super().__init__()
        self.lconv_coefficient = lconv_coefficient

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight * self.lconv_coefficient


class LConvNormHook:

    """
    Kernel normalization for Lipschitz convolution. Normalize weights
    based on input shape and kernel size, see https://arxiv.org/abs/2006.06520
    """

    def apply(self, module: torch.nn.Module, name: str = "weight") -> None:
        self.name = name
        self.coefficient = None

        if not isinstance(module, torch.nn.Conv2d):
            raise RuntimeError(
                "Can only apply lconv_norm hooks on 2D-convolutional layer."
            )

        module.register_forward_pre_hook(self)

    def __call__(self, module: torch.nn.Conv2d, inputs: Any):
        coefficient = compute_lconv_coef(
            module.kernel_size, inputs[0].shape[-4:], module.stride
        )
        # the parametrization is updated only if the coefficient has changed
        if coefficient != self.coefficient:
            if hasattr(module, "parametrizations"):
                self.remove_parametrization(module)
            parametrize.register_parametrization(
                module, self.name, _LConvNorm(coefficient)
            )
            self.coefficient = coefficient

    def remove_parametrization(self, module: nn.Module) -> nn.Module:
        r"""
        Removes the normalization reparameterization from a module.

        Args:
            module: Containing module.

        Example:
            >>> m = bjorck_norm(nn.Linear(20, 40))
            >>> remove_bjorck_norm(m)
        """
        for key, m in module.parametrizations[self.name]._modules.items():
            if isinstance(m, _LConvNorm):
                if len(module.parametrizations[self.name]) == 1:
                    parametrize.remove_parametrizations(module, self.name)
                else:
                    del module.parametrizations[self.name]._modules[key]


def lconv_norm(module: torch.nn.Conv2d, name: str = "weight") -> torch.nn.Conv2d:
    r"""
    Applies Lipschitz normalization to a kernel in the given convolutional.
    This is implemented via a hook that multiplies the kernel by a value computed
    from the input shape before every ``forward()`` call.

    See `Achieving robustness in classification using optimal transport with hinge
    regularization <https://arxiv.org/abs/2006.06520>`_.

    Args:
        module: Containing module.
        name: Name of weight parameter.

    Returns:
        The original module with the Lipschitz normalization hook.

    Example::

        >>> m = lconv_norm(nn.Conv2d(16, 16, (3, 3)))
        >>> m
        Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))

    """
    LConvNormHook().apply(module, name)
    return module


def remove_lconv_norm(module: torch.nn.Conv2d) -> torch.nn.Conv2d:
    r"""
    Removes the Lipschitz normalization hook from a module.

    Args:
        module: Containing module.

    Example:

        >>> m = lconv_norm(nn.Conv2d(16, 16, (3, 3)))
        >>> remove_lconv_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, LConvNormHook):
            hook.remove_parametrization(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("lconv_norm not found in {}".format(module))
