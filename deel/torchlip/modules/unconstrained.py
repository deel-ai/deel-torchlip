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

from typing import Union
import torch
from torch.nn.common_types import _size_1_t, _size_2_t
from ..functional import SymmetricPad


class PadConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        """
        This class is a Conv1d Layer with additional padding modes

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

        This documentation reuse the body of the original torch.nn.Conv1d doc.
        """

        self.old_padding = padding
        self.old_padding_mode = padding_mode
        if padding_mode.lower() == "symmetric":
            padding_mode = "zeros"
            padding = "valid"

        super(PadConv1d, self).__init__(
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

        if self.old_padding_mode.lower() == "symmetric":
            self.pad = SymmetricPad(self.old_padding, onedim=True)
        else:
            self.pad = lambda x: x

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super(PadConv1d, self).forward(self.pad(input))

    def vanilla_export(self):
        if self.old_padding_mode.lower() == "symmetric":
            next_layer_type = PadConv1d
        else:
            next_layer_type = torch.nn.Conv1d

        layer = next_layer_type(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.old_padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
            padding_mode=self.old_padding_mode,
        )
        layer.weight.data = self.weight.detach()
        if self.bias is not None:
            layer.bias.data = self.bias.detach()
        return layer


class PadConv2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        """
        This class is a Conv2d Layer with additional padding modes

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

        This documentation reuse the body of the original torch.nn.Conv2D doc.
        """

        self.old_padding = padding
        self.old_padding_mode = padding_mode
        if padding_mode.lower() == "symmetric":
            # symmetric padding of one pixel can be replaced by replicate
            if (isinstance(padding, int) and padding <= 1) or (
                isinstance(padding, tuple) and padding[0] <= 1 and padding[1] <= 1
            ):
                self.old_padding_mode = padding_mode = "replicate"
            else:
                padding_mode = "zeros"
                padding = "valid"

        super(PadConv2d, self).__init__(
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

        if self.old_padding_mode.lower() == "symmetric":
            self.pad = SymmetricPad(self.old_padding)
        else:
            self.pad = lambda x: x

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super(PadConv2d, self).forward(self.pad(input))

    def vanilla_export(self):
        if self.old_padding_mode.lower() == "symmetric":
            next_layer_type = PadConv2d
        else:
            next_layer_type = torch.nn.Conv2d

        layer = next_layer_type(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.old_padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
            padding_mode=self.old_padding_mode,
        )
        layer.weight.data = self.weight.detach()
        if self.bias is not None:
            layer.bias.data = self.bias.detach()
        return layer
