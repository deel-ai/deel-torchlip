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
            kernel_size: the size of the window
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            ceil_mode: when True, will use `ceil` instead of `floor` to compute
            the output shape
            count_include_pad: when True, will include the zero-padding in the
            averaging calculation
            divisor_override: if specified, it will be used as divisor,
            otherwise :attr:`kernel_size` will be used
            k_coef_lip: the lipschitz factor to ensure

        This documentation reuse the body of the original torch.nn.AveragePooling2D
        doc.
        """
        if stride is not None:
            raise RuntimeError("stride must be equal to pool_size.")
        if padding != 0:
            raise RuntimeError(f"{type(self)} does not support padding.")
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = self._coefficient_lip * math.sqrt(np.prod(np.asarray(self.kernel_size)))
        return torch.nn.AvgPool2d.forward(self, input) * coeff  # type: ignore

    def vanilla_export(self):
        return self


class ScaledGlobalAvgPool2d(torch.nn.AdaptiveAvgPool2d, LipschitzModule):
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
            output_size: the target output size of the image of the form H x W.
                        Can be a tuple (H, W) or a single H for a square image H x H.
                        H and W can be either a ``int``, or ``None`` which means the
                        size will be the same as that of the input.
            return_indices: if ``True``, will return the indices along with the outputs.
                            Useful to pass to nn.MaxUnpool2d. Default: ``False``

        This documentation reuse the body of the original
        nn.AdaptiveAvgPool2d doc.
        """
        torch.nn.AdaptiveAvgPool2d.__init__(self, output_size)
        LipschitzModule.__init__(self, k_coef_lip)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = math.sqrt(input.shape[-2] * input.shape[-1]) * self._coefficient_lip

        return torch.nn.AdaptiveAvgPool2d.forward(self, input) * coeff


class ScaledL2NormPooling2D(torch.nn.AvgPool2d, LipschitzModule):
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
        pooling operation is norm preserving (aka gradient=1 almost everywhere).

        [1]Y.-L.Boureau, J.Ponce, et Y.LeCun, « A Theoretical Analysis of Feature
        Pooling in Visual Recognition »,p.8.

        Arguments:
            pool_size: integer or tuple of 2 integers,
                factors by which to downscale (vertical, horizontal).
                `(2, 2)` will halve the input in both spatial dimension.
                If only one integer is specified, the same window length
                will be used for both dimensions.
            strides: Integer, tuple of 2 integers, or None.
                Strides values.
                If None, it will default to `pool_size`.
            padding: One of `"valid"` or `"same"` (case-insensitive).
            k_coef_lip: the lipschitz factor to ensure
            eps_grad_sqrt: Epsilon value to avoid numerical instability
                due to non-defined gradient at 0 in the sqrt function

        Input shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, rows, cols, channels)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, channels, rows, cols)`.

        Output shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
        """
        if stride is not None:
            raise RuntimeError("stride must be equal to pool_size")
        if padding != 0:
            raise RuntimeError("ScaledAveragePooling2D only support padding='valid'")
        if eps_grad_sqrt < 0.0:
            raise RuntimeError("eps_grad_sqrt must be positive")
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = self._coefficient_lip * math.sqrt(np.prod(np.asarray(self.kernel_size)))
        return (  # type: ignore
            sqrt_with_gradeps(
                torch.nn.AvgPool2d.forward(self, torch.square(input)),
                self.eps_grad_sqrt,
            )
            * coeff
        )
