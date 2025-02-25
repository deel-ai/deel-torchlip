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
import numpy as np
import torch
from torch.nn.common_types import _size_2_t
from torch.nn.utils.parametrizations import spectral_norm

from ..utils import bjorck_norm
from ..normalizers import DEFAULT_EPS_BJORCK
from ..normalizers import DEFAULT_EPS_SPECTRAL
from ..utils import frobenius_norm
from ..utils import lconv_norm
from .unconstrained import PadConv1d, PadConv2d
from .module import LipschitzModule


class SpectralConv1d(PadConv1d, LipschitzModule):
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
        eps_spectral: int = DEFAULT_EPS_SPECTRAL,
        eps_bjorck: int = DEFAULT_EPS_BJORCK,
    ):
        """
        This class is a Conv1d Layer constrained such that all singular of it's kernel
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
            eps_spectral: stopping criterion for the iterative power algorithm.
            eps_bjorck: stopping criterion Bjorck algorithm.

        This documentation reuse the body of the original torch.nn.Conv1D doc.
        """

        PadConv1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
        )
        LipschitzModule.__init__(self, k_coef_lip)

        torch.nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

        spectral_norm(
            self,
            name="weight",
            eps=eps_spectral,
        )
        bjorck_norm(self, name="weight", eps=eps_bjorck)
        lconv_norm(self)
        self.apply_lipschitz_factor()

    def vanilla_export(self):
        return PadConv1d.vanilla_export(self)


class SpectralConv2d(PadConv2d, LipschitzModule):
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
        eps_spectral: int = DEFAULT_EPS_SPECTRAL,
        eps_bjorck: int = DEFAULT_EPS_BJORCK,
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
                ``'replicate'``, ``'symmetric'``  or ``'circular'``.
                Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel elements.
                Has to be one
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Has to be one
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output.
            k_coef_lip: Lipschitz constant to ensure.
            eps_spectral: stopping criterion for the iterative power algorithm.
            eps_bjorck: stopping criterion Bjorck algorithm.

        This documentation reuse the body of the original torch.nn.Conv2D doc.
        """

        PadConv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
        )
        LipschitzModule.__init__(self, k_coef_lip)

        torch.nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

        spectral_norm(
            self,
            name="weight",
            eps=eps_spectral,
        )
        bjorck_norm(self, name="weight", eps=eps_bjorck)
        lconv_norm(self, name="weight")
        self.apply_lipschitz_factor()

    def vanilla_export(self):
        return PadConv2d.vanilla_export(self)


class FrobeniusConv2d(PadConv2d, LipschitzModule):
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

        PadConv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
            dilation=dilation,
            groups=groups,
        )
        LipschitzModule.__init__(self, k_coef_lip)

        torch.nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

        frobenius_norm(self, name="weight", disjoint_neurons=False)
        lconv_norm(self)
        self.apply_lipschitz_factor()

    def vanilla_export(self):
        return PadConv2d.vanilla_export(self)


class SpectralConvTranspose2d(torch.nn.ConvTranspose2d, LipschitzModule):
    r"""Applies a 2D transposed convolution operator over an input image
    such that all singular of it's kernel are 1.
    The computation are the same as for SpectralConv2d layer

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution.
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input.
        output_padding: only 0 or none are supported
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements.
            Has to be one.
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Has to be one.
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output.
        k_coef_lip: Lipschitz constant to ensure.
        eps_spectral: stopping criterion for the iterative power algorithm.
        eps_bjorck: stopping criterion Bjorck algorithm.

    This documentation reuse the body of the original torch.nn.ConvTranspose2d
    doc.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        k_coef_lip: float = 1.0,
        eps_spectral: int = DEFAULT_EPS_SPECTRAL,
        eps_bjorck: int = DEFAULT_EPS_BJORCK,
    ) -> None:
        if dilation != 1:
            raise ValueError("SpectralConvTranspose2d does not support dilation rate")
        if output_padding not in [0, None]:
            raise ValueError("SpectralConvTranspose2d only supports output_padding=0")
        torch.nn.ConvTranspose2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        LipschitzModule.__init__(self, k_coef_lip)

        torch.nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

        spectral_norm(
            self,
            name="weight",
            eps=eps_spectral,
        )
        bjorck_norm(self, name="weight", eps=eps_bjorck)
        lconv_norm(self, name="weight")
        self.apply_lipschitz_factor()

    def vanilla_export(self):
        layer = torch.nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        layer.weight.data = self.weight.detach()
        if self.bias is not None:
            layer.bias.data = self.bias.detach()
        return layer
