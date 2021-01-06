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

        if self.stride != self.kernel_size:
            raise RuntimeError("stride must be equal to kernel_size.")
        if np.sum(self.padding) != 0:
            raise RuntimeError(f"{type(self)} does not support padding.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = self._coefficient_lip * math.sqrt(np.prod(np.asarray(self.kernel_size)))
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

        if self.stride != self.kernel_size:
            raise RuntimeError("stride must be equal to kernel_size.")
        if np.sum(self.padding) != 0:
            raise RuntimeError("ScaledL2NormPooling2D does not support padding.")
        if eps_grad_sqrt < 0.0:
            raise RuntimeError("eps_grad_sqrt must be positive")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = self._coefficient_lip * math.sqrt(np.prod(np.asarray(self.kernel_size)))
        return (  # type: ignore
            sqrt_with_gradeps(
                torch.nn.AvgPool2d.forward(self, torch.square(input)),
                self.eps_grad_sqrt,
            )
            * coeff
        )
