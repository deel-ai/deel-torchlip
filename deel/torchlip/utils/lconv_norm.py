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

from .hook_norm import HookNorm


def compute_lconv_coef(
    kernel_size: Tuple[int, ...],
    input_shape: Tuple[int, ...],
    strides: Tuple[int, ...] = (1, 1),
    padding_mode: str = "zeros"
) -> float:
    # See https://arxiv.org/abs/2006.06520
    stride = np.prod(strides)
    k1, k2 = kernel_size
    h, w = input_shape[-2:]
    if padding_mode == "replicate" or padding_mode == "circular" or padding_mode == "reflect":
        #print(padding_mode)
        coefLip = float(strides[0])/float(k1)
    elif stride == 1:
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
    #print(coefLip)
    return coefLip  # type: ignore


def compute_lconv_coef_1d(
    kernel_size,
    input_shape,
    strides,
    padding_mode
) -> float:
    # See https://arxiv.org/abs/2006.06520
    return 1.  #à faire


class LConvNorm(HookNorm):

    """
    Kernel normalization for Lipschitz convolution. Normalize weights
    based on input shape and kernel size, see https://arxiv.org/abs/2006.06520
    """

    @staticmethod
    def apply(module: torch.nn.Module) -> "LConvNorm":

        if not isinstance(module, (torch.nn.Conv2d,torch.nn.Conv1d)):
            raise RuntimeError(
                "Can only apply lconv_norm hooks on 2D-convolutional layer."
            )

        return LConvNorm(module, "weight")

    def compute_weight(self, module: torch.nn.Module, inputs: Any) -> torch.Tensor:
        assert isinstance(module, (torch.nn.Conv2d,torch.nn.Conv1d))
        if isinstance(module, torch.nn.Conv1d):
            coefficient = compute_lconv_coef_1d(
                module.kernel_size, inputs[0].shape[-4:], module.stride,module.padding_mode
            )
        else:
            coefficient = compute_lconv_coef(
                module.kernel_size, inputs[0].shape[-4:], module.stride,module.padding_mode
            )
        return self.weight(module) * coefficient


def lconv_norm(module) :
    r"""
    Applies Lipschitz normalization to a kernel in the given convolutional.
    This is implemented via a hook that multiplies the kernel by a value computed
    from the input shape before every ``forward()`` call.

    See `Achieving robustness in classification using optimal transport with hinge
    regularization <https://arxiv.org/abs/2006.06520>`_.

    Args:
        module: Containing module.

    Returns:
        The original module with the Lipschitz normalization hook.

    Example::

        >>> m = lconv_norm(nn.Conv2d(16, 16, (3, 3)))
        >>> m
        Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))

    """
    LConvNorm.apply(module)
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
        if isinstance(hook, LConvNorm):
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("lconv_norm not found in {}".format(module))
