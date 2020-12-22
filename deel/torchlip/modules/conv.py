# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

import numpy as np

import torch

from torch.nn.utils import spectral_norm
from torch.nn.common_types import _size_2_t

from ..utils import (
    DEFAULT_NITER_BJORCK,
    DEFAULT_NITER_SPECTRAL,
    bjorck_norm,
    frobenius_norm,
    lconv_norm,
)
from .module import LipschitzModule


class SpectralConv2d(torch.nn.Conv2d, LipschitzModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        k_coef_lip: float = 1.0,
        niter_spectral: int = DEFAULT_NITER_SPECTRAL,
        niter_bjorck: int = DEFAULT_NITER_BJORCK,
    ):
        """
        This class is a Conv2d Layer constrained such that all singular of it's kernel
        are 1. The computation based on BjorckNormalizer algorithm. As this is not
        enough to ensure 1-Lipschitz a coercive coefficient is applied on the
        output.
        The computation is done in three steps:

        1. reduce the largest singular value to 1, using iterated power method.
        2. increase other singular values to 1, using BjorckNormalizer algorithm.
        3. divide the output by the Lipschitz bound to ensure k-Lipschitz.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution.
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input.
            padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel elements.
            groups (int, optional): Number of blocked connections from input
                channels to output channels.
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output.
            k_coef_lip: Lipschitz constant to ensure.
            niter_spectral: Number of iteration to find the maximum singular value.
            niter_bjorck: Number of iteration with BjorckNormalizer algorithm.

        This documentation reuse the body of the original torch.nn.Conv2D doc.
        """
        # if not ((dilation == (1, 1)) or (dilation == [1, 1]) or (dilation == 1)):
        #     raise RuntimeError("NormalizedConv does not support dilation rate")
        # if padding_mode != "same":
        #     raise RuntimeError("NormalizedConv only support padding='same'")

        torch.nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        )
        LipschitzModule.__init__(self, k_coef_lip)
        spectral_norm(
            self,
            name="weight",
            n_power_iterations=niter_spectral,
        )
        bjorck_norm(self, name="weight", n_iterations=niter_bjorck)
        lconv_norm(self)
        self.register_forward_pre_hook(self._hook)

    def vanilla_export(self):
        layer = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        layer.weight.data = self.weight.detach()
        if self.bias is not None:
            layer.bias.data = self.bias.detach()
        return layer


class FrobeniusConv2d(torch.nn.Conv2d, LipschitzModule):
    """
    Same as SpectralConv2d but in the case of a single output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        k_coef_lip: float = 1.0,
    ):
        if np.prod([stride]) != 1:
            raise RuntimeError("FrobeniusConv2d does not support strides")
        # if padding_mode != "same":
        #     raise RuntimeError("NormalizedConv only support padding='same'")

        torch.nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        LipschitzModule.__init__(self, k_coef_lip)

        frobenius_norm(self, name="weight")
        lconv_norm(self)
        self.register_forward_pre_hook(self._hook)

    def vanilla_export(self):
        layer = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        layer.weight.data = self.weight.detach()
        if self.bias is not None:
            layer.bias.data = self.bias.detach()
        return layer
