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
import math
from typing import Optional

import numpy as np
import torch
from torch.nn.common_types import _size_2_t

from ..utils import sqrt_with_gradeps
from .module import LipschitzModule


def computePoolScalingFactor(kernel_size):
    if isinstance(kernel_size, tuple):
        scalingFactor = math.sqrt(np.prod(np.asarray(kernel_size)))
    else:
        scalingFactor = kernel_size
    return scalingFactor


class ScaledAvgPool2d(torch.nn.AvgPool2d, LipschitzModule):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: bool = None,
        k_coef_lip: float = 1.0,
    ):
        """
        Average pooling operation for spatial data, but with a lipschitz bound.

        Args:
            kernel_size: The size of the window.
            stride: The stride of the window. Must be None or equal to
                ``kernel_size``. Default value is ``kernel_size``.
            padding: Implicit zero-padding to be added on both sides. Must
                be zero.
            ceil_mode: When True, will use ceil instead of floor to compute the output
                shape.
            count_include_pad: When True, will include the zero-padding in the averaging
                calculation.
            divisor_override: If specified, it will be used as divisor, otherwise
                ``kernel_size`` will be used.
            k_coef_lip: The Lipschitz factor to ensure. The output will be scaled
                by this factor.

        This documentation reuse the body of the original torch.nn.AveragePooling2D
        doc.
        """
        torch.nn.AvgPool2d.__init__(
            self,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        LipschitzModule.__init__(self, k_coef_lip)

        self.scalingFactor = computePoolScalingFactor(self.kernel_size)

        if self.stride != self.kernel_size:
            raise RuntimeError("stride must be equal to kernel_size.")
        if np.sum(self.padding) != 0:
            raise RuntimeError(f"{type(self)} does not support padding.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = self._coefficient_lip * self.scalingFactor
        return torch.nn.AvgPool2d.forward(self, input) * coeff

    def vanilla_export(self):
        return self


class ScaledAdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d, LipschitzModule):
    def __init__(
        self,
        output_size: _size_2_t,
        k_coef_lip: float = 1.0,
    ):
        """
        Applies a 2D adaptive max pooling over an input signal composed of several
        input planes.

        The output is of size H x W, for any input size.
        The number of output features is equal to the number of input planes.

        Args:
            output_size: The target output size of the image of the form H x W.
                Can be a tuple (H, W) or a single H for a square image H x H.
                H and W can be either a ``int``, or ``None`` which means the
                size will be the same as that of the input.
            k_coef_lip: The Lipschitz factor to ensure. The output will be scaled
                by this factor.

        This documentation reuse the body of the original
        nn.AdaptiveAvgPool2d doc.
        """
        torch.nn.AdaptiveAvgPool2d.__init__(self, output_size)
        LipschitzModule.__init__(self, k_coef_lip)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = math.sqrt(input.shape[-2] * input.shape[-1]) * self._coefficient_lip
        return torch.nn.AdaptiveAvgPool2d.forward(self, input) * coeff

    def vanilla_export(self):
        return self


class ScaledL2NormPool2d(torch.nn.AvgPool2d, LipschitzModule):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: bool = None,
        k_coef_lip: float = 1.0,
        eps_grad_sqrt: float = 1e-6,
    ):
        """
        Average pooling operation for spatial data, with a lipschitz bound. This
        pooling operation is norm preserving (gradient=1 almost everywhere).

        [1] Y.-L.Boureau, J.Ponce, et Y.LeCun, « A Theoretical Analysis of Feature
        Pooling in Visual Recognition »,p.8.

        Args:
            kernel_size: The size of the window.
            stride: The stride of the window. Must be None or equal to
                ``kernel_size``. Default value is ``kernel_size``.
            padding: Implicit zero-padding to be added on both sides. Must
                be zero.
            ceil_mode: When True, will use ceil instead of floor to compute the output
                shape.
            count_include_pad: When True, will include the zero-padding in the averaging
                calculation.
            divisor_override: If specified, it will be used as divisor, otherwise
                ``kernel_size`` will be used.
            k_coef_lip: The lipschitz factor to ensure. The output will be
                scaled by this factor.
            eps_grad_sqrt: Epsilon value to avoid numerical instability
                due to non-defined gradient at 0 in the sqrt function
        """
        torch.nn.AvgPool2d.__init__(
            self,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        LipschitzModule.__init__(self, k_coef_lip)
        self.eps_grad_sqrt = eps_grad_sqrt
        self.scalingFactor = computePoolScalingFactor(self.kernel_size)

        if self.stride != self.kernel_size:
            raise RuntimeError("stride must be equal to kernel_size.")
        if np.sum(self.padding) != 0:
            raise RuntimeError("ScaledL2NormPooling2D does not support padding.")
        if eps_grad_sqrt < 0.0:
            raise RuntimeError("eps_grad_sqrt must be positive")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = self._coefficient_lip * self.scalingFactor
        return (  # type: ignore
            sqrt_with_gradeps(
                torch.nn.AvgPool2d.forward(self, torch.square(input)),
                self.eps_grad_sqrt,
            )
            * coeff
        )

    def vanilla_export(self):
        return self


class ScaledGlobalL2NormPool2d(torch.nn.AdaptiveAvgPool2d, LipschitzModule):
    def __init__(
        self,
        output_size: _size_2_t = (1, 1),
        k_coef_lip: float = 1.0,
        eps_grad_sqrt: float = 1e-6,
    ):
        """
        Average pooling operation for spatial data, with a lipschitz bound. This
        pooling operation is norm preserving (aka gradient=1 almost everywhere).

        [1]Y.-L.Boureau, J.Ponce, et Y.LeCun, « A Theoretical Analysis of Feature
        Pooling in Visual Recognition »,p.8.

        Arguments:
            output_size: the target output size has to be (1,1)
            k_coef_lip: the lipschitz factor to ensure
            eps_grad_sqrt: Epsilon value to avoid numerical instability
                due to non-defined gradient at 0 in the sqrt function

        Input shape:
            4D tensor with shape `(batch_size, channels, rows, cols)`.

        Output shape:
            4D tensor with shape `(batch_size, channels, 1, 1)`.
        """
        if eps_grad_sqrt < 0.0:
            raise RuntimeError("eps_grad_sqrt must be positive")
        if not isinstance(output_size, tuple) or len(output_size) != 2:
            raise RuntimeError("output_size must be a tuple of 2 integers")
        else:
            if output_size[0] != 1 or output_size[1] != 1:
                raise RuntimeError("output_size must be (1, 1)")
        torch.nn.AdaptiveAvgPool2d.__init__(self, output_size)
        LipschitzModule.__init__(self, k_coef_lip)
        self.eps_grad_sqrt = eps_grad_sqrt

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # coeff = computePoolScalingFactor(input.shape[-2:]) * self._coefficient_lip
        # avg = torch.nn.AdaptiveAvgPool2d.forward(self, torch.square(input))
        # return  sqrt_with_gradeps(avg,self.eps_grad_sqrt)* coeff
        return (  # type: ignore
            sqrt_with_gradeps(
                torch.square(input).sum(axis=(2, 3), keepdim=True),
                self.eps_grad_sqrt,
            )
            * self._coefficient_lip
        )

    def vanilla_export(self):
        return self
