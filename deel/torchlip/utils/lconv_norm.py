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
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


def compute_lconv_coef_1d(
    kernel_size: Tuple[int],
    input_shape: Tuple[int] = None,
    strides: Tuple[int] = (1,),
    padding_mode: str = "zeros",
) -> float:
    stride = strides[0]
    k1 = kernel_size[0]

    if (
        (padding_mode in ["zeros", "same"])
        and (stride == 1)
        and (input_shape is not None)
    ):
        # See https://arxiv.org/abs/2006.06520
        in_l = input_shape[-1]
        k1_div2 = (k1 - 1) / 2
        coefLip = in_l / (k1 * in_l - k1_div2 * (k1_div2 + 1))
    else:
        sn1 = strides[0]
        coefLip = 1.0 / np.ceil(k1 / sn1)

    return coefLip  # type: ignore


def compute_lconv_coef(
    kernel_size: Tuple[int, ...],
    input_shape: Tuple[int, ...] = None,
    strides: Tuple[int, ...] = (1, 1),
    padding_mode: str = "zeros",
) -> float:
    # See https://arxiv.org/abs/2006.06520
    stride = np.prod(strides)
    k1, k2 = kernel_size

    if (
        (padding_mode in ["zeros", "same"])
        and (stride == 1)
        and (input_shape is not None)
    ):
        h, w = input_shape[-2:]
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
    """Parametrization module for kernel normalization of lipschitz convolution."""

    def __init__(self, lconv_coefficient: float) -> None:
        super().__init__()
        self.lconv_coefficient = lconv_coefficient

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight * self.lconv_coefficient


ConvType = Union[torch.nn.Conv2d, torch.nn.Conv1d]


def lconv_norm(module: ConvType, name: str = "weight") -> ConvType:
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
    onedim = isinstance(module, torch.nn.Conv1d)
    if onedim:
        coefficient = compute_lconv_coef_1d(module.kernel_size, None, module.stride)
    else:
        coefficient = compute_lconv_coef(module.kernel_size, None, module.stride)
    parametrize.register_parametrization(module, name, _LConvNorm(coefficient))
    return module


def remove_lconv_norm(module: torch.nn.Conv2d, name: str = "weight") -> torch.nn.Conv2d:
    r"""
    Removes the normalization parametrization for lipschitz convolution from a module.

    Args:
        module: Containing module.
        name: Name of weight parameter.

    Example:

        >>> m = lconv_norm(nn.Conv2d(16, 16, (3, 3)))
        >>> remove_lconv_norm(m)
    """
    for key, m in module.parametrizations[name]._modules.items():
        if isinstance(m, _LConvNorm):
            if len(module.parametrizations[name]) == 1:
                parametrize.remove_parametrizations(module, name)
            else:
                del module.parametrizations[name]._modules[key]
