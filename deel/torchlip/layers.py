# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module extends original Pytorch layers, in order to add k lipschitz constraint via
reparametrization. Currently, are implemented:

* Linear layer:
    as SpectralLinear
* Conv1d layer:
    as SpectralConv1d
* Conv2d layer:
    as SpectralConv2d
* Conv3d layer:
    as SpectralConv3d
* AvgPool2d:
    as ScaledAvgPool2d

By default the layers are 1 Lipschitz almost everywhere, which is efficient for
wasserstein distance estimation. However for other problems (such as adversarial
robustness) the user may want to use layers that are at most 1 lipschitz, this can
be done by setting the param `niter_bjorck=0`.
"""

import abc
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn.utils import spectral_norm
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional
from torch.nn.modules.utils import _single

from .normalizers import bjorck_normalization
from .init import spectral_

from .utils import (
    DEFAULT_NITER_BJORCK,
    DEFAULT_NITER_SPECTRAL,
    compute_lconv_ip_coef,
)


class LipschitzLayer(abc.ABC):
    """
    This class allow to set lipschitz factor of a layer. Lipschitz layer must inherit
    this class to allow user to set the lipschitz factor.

    Warning:
         This class only regroup useful functions when developing new Lipschitz layers.
         But it does not ensure any property about the layer. This means that
         inheriting from this class won't ensure anything about the lipschitz constant.
    """

    k_coef_lip = 1.0
    """variable used to store the lipschitz factor"""
    coef_lip = None
    """
    define correction coefficient (ie. Lipschitz bound ) of the layer
    ( multiply the output of the layer by this constant )
    """
    is_init = False
    """variable used to initialize the lipschitz factor  """

    def set_klip_factor(self, klip_factor):
        """
        Allow to set the lipschitz factor of a layer.

        Args:
            klip_factor: the Lipschitz factor the user want to ensure.

        Returns:
            None

        """
        self.k_coef_lip = klip_factor

    @abc.abstractmethod
    def _compute_lip_coef(self, input_shape=None):
        """
        Some layers (like convolution) cannot ensure a strict lipschitz constant (as
        the Lipschitz factor depends on the input data). Those layers then rely on the
        computation of a bounding factor. This function allow to compute this factor.

        Args:
            input_shape: the shape of the input of the layer.

        Returns:
            the bounding factor.

        """
        pass

    def _init_lip_coef(self, input_shape):
        """
        Initialize the lipschitz coefficient of a layer.

        Args:
            input_shape: the layers input shape

        Returns:
            None

        """
        self.coef_lip = self._compute_lip_coef(input_shape)

    def _get_coef(self):
        """
        Returns:
            the multiplicative coefficient to be used on the result in order to ensure
            k-Lipschitzity.
        """
        if self.coef_lip is None:
            raise RuntimeError("compute_coef must be called before calling get_coef")
        return self.coef_lip * self.k_coef_lip


class Condensable(abc.ABC):
    """
    Some Layers don't optimize directly the kernel, this means that the kernel stored
    in the layer is not the kernel used to make predictions (called W_bar), to address
    this, these layers can implement the condense() function that make self.kernel equal
    to W_bar.

    This operation also allow the turn the lipschitz layer to it torch equivalent ie.
    The Dense layer that have the same predictions as the trained SpectralDense.
    """

    def condense(self):
        """
        The condense operation allow to overwrite the kernel and ensure that other
        variables are still consistent.
        The stored kernel is W_bar used to make predictions in spectral layers,
        this method is not implemented

        Returns:
            None

        """
        pass

    @abc.abstractmethod
    def vanilla_export(self):
        """
        This operation allow to turn this Layer to it's super type, easing storage and
        serving.

        Returns:
             self as super type

        """
        pass


class SpectralLinear(nn.Linear, LipschitzLayer, Condensable):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        k_coef_lip: float = 1.0,
        niter_spectral: int = DEFAULT_NITER_SPECTRAL,
        niter_bjorck: int = DEFAULT_NITER_BJORCK,
    ):
        """
        This class is a Linear Layer constrained such that all singular of it's kernel
        are 1. The computation based on BjorckNormalizer algorithm.
        The computation is done in two steps:

        1. reduce the larget singular value to 1, using iterated power method.
        2. increase other singular values to 1, using BjorckNormalizer algorithm.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If ``False``, the layer will not learn an additive bias.
            k_coef_lip: Lipschitz constant to ensure.
            niter_spectral: Number of iteration to find the maximum singular value.
            niter_bjorck: Number of iteration with BjorckNormalizer algorithm.

        Shape:
            - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
            additional dimensions and :math:`H_{in} = \\text{in\\_features}`
            - Output: :math:`(N, *, H_{out})` where all but the last dimension
            are the same shape as the input and :math:`H_{out} = \\text{out\\_features}`..

        This documentation reuse the body of the original torch.nn.Linear doc.
        """
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        self.niter_spectral = niter_spectral
        self.niter_bjorck = niter_bjorck
        self.k_coef_lip = k_coef_lip
        self.set_klip_factor(self.k_coef_lip)
        self._init_lip_coef(None)
        spectral_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
        # spectral normalization is performed during forward.
        # spectral_norm is implemented via a hook that calculates
        # spectral norm and rescales weight before every :meth:~Module.forward call.
        spectral_norm(
            self,
            name="weight",
            n_power_iterations=self.niter_spectral,
        )

    def _compute_lip_coef(self, input_shape=None):
        return 1.0  # this layer don't require a corrective factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        W_bar = bjorck_normalization(self.weight, niter=self.niter_bjorck)
        return F.linear(input, W_bar * self._get_coef(), self.bias)

    def condense(self):
        bjorck_normalization(self.weight, niter=self.niter_bjorck)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias,
            **self._kwargs
        )
        layer.weight = self.weight * self._get_coef()
        if self.bias:
            layer.bias = self.bias
        return layer

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "niter_spectral": self.niter_spectral,
            "niter_bjorck": self.niter_bjorck,
        }
        base_config = super().state_dict(destination, prefix, keep_vars)
        return dict(list(base_config.items()) + list(config.items()))


class FrobeniusLinear(nn.Linear, LipschitzLayer, Condensable):
    """
    Same a SpectralLinear, but in the case of a single output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        k_coef_lip: float = 1.0,
        niter_spectral: int = DEFAULT_NITER_SPECTRAL,
    ):
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        self.niter_spectral = niter_spectral
        self.k_coef_lip = k_coef_lip
        self._init_lip_coef(None)
        self.set_klip_factor(k_coef_lip)
        init.orthogonal_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        W_bar = (
            self.weight
            / np.linalg.norm(self.weight.detach().numpy())
            * self._get_coef()
        )
        return F.linear(input, W_bar, self.bias)

    def condense(self):
        W_bar = self.weight.data / np.linalg.norm(self.weight.detach().numpy())
        self.weight.data = W_bar

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = nn.Linear(
            bias=self.bias,
            in_features=self.in_features,
            out_features=self.out_features,
        )
        layer.reset_parameters(self)
        layer.weight.data = self.weight * self._get_coef()
        if self.bias is not None:
            layer.bias.data = self.bias.data
        return layer

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "niter_spectral": self.niter_spectral,
        }
        base_config = super().state_dict(destination, prefix, keep_vars)
        return dict(list(base_config.items()) + list(config.items()))


class SpectralConv1d(nn.Conv1d, LipschitzLayer, Condensable):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        k_coef_lip: float = 1.0,
        niter_spectral: int = DEFAULT_NITER_SPECTRAL,
        niter_bjorck: int = DEFAULT_NITER_BJORCK,
    ):
        """
        This class is a Conv1d Layer constrained such that all singular of it's kernel
        are 1. The computation based on BjorckNormalizer algorithm. As this is not
        enough to ensure 1 Lipschitzity a coertive coefficient is applied on the
        output.
        The computation is done in three steps:

        1. reduce the largest singular value to 1, using iterated power method.
        2. increase other singular values to 1, using BjorckNormalizer algorithm.
        3. divide the output by the Lipschitz bound to ensure k Lipschitzity.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 0
            padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel
                elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Default: ``True``
            k_coef_lip: lipschitz constant to ensure
            niter_spectral: number of iteration to find the maximum singular value.
            niter_bjorck: number of iteration with BjorckNormalizer algorithm.

        This documentation reuse the body of the original torch.nn.Conv2D doc.
        """

        # TODO: Not the same system between torch and tensorflow. Needs to check the
        # value of "padding" or someting like that (and maybe "padding_mode").
        # if padding_mode != "same":
        #     raise RuntimeError("NormalizedConv only support padding='same'")

        if niter_spectral < 1:
            raise RuntimeError("niter_spectral has to be > 0")

        nn.Conv1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.niter_spectral = niter_spectral
        self.niter_bjorck = niter_bjorck
        self.k_coef_lip = k_coef_lip
        self.set_klip_factor(self.k_coef_lip)
        spectral_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
        # spectral normalization is performed during forward.
        # spectral_norm is implemented via a hook that calculates
        # spectral norm and rescales weight before every :meth:~Module.forward call.
        spectral_norm(
            self,
            name="weight",
            n_power_iterations=self.niter_spectral,
        )

    def _compute_lip_coef(self, input_shape=None):
        # According to the file lipschitz_CNN.pdf
        return compute_lconv_ip_coef(self.kernel_size, input_shape, self.stride)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # TODO: Ça doit pouvoir se faire dans le __init__().
        if not self.is_init:
            self._init_lip_coef(input.shape[1:])
            self.is_init = True

        W_bar = bjorck_normalization(self.weight, niter=self.niter_bjorck)
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    input,
                    self._reversed_padding_repeated_twice,  # type: ignore
                    mode=self.padding_mode,
                ),
                W_bar * self._get_coef(),
                self.bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            input,
            W_bar * self._get_coef(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def condense(self):
        bjorck_normalization(self.weight, niter=self.niter_bjorck)

    def vanilla_export(self):
        layer = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            padding_mode=self.padding_mode,
        )

        # TODO: Does this work? I don't think we should use setattr() here,
        # there are proper ways to set weights on module in torch.
        layer.reset_parameters(self)
        setattr(self, "weight", self.weight * self._get_coef())
        if self.bias:
            setattr(self, "bias", self.bias)
        return layer

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "niter_spectral": self.niter_spectral,
            "niter_bjorck": self.niter_bjorck,
        }
        base_config = super().state_dict(destination, prefix, keep_vars)
        return dict(list(base_config.items()) + list(config.items()))


class SpectralConv2d(nn.Conv2d, LipschitzLayer, Condensable):
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
        padding_mode: str = "same",
        k_coef_lip: float = 1.0,
        niter_spectral: int = DEFAULT_NITER_SPECTRAL,
        niter_bjorck: int = DEFAULT_NITER_BJORCK,
    ):
        """
        This class is a Conv2d Layer constrained such that all singular of it's kernel
        are 1. The computation based on BjorckNormalizer algorithm. As this is not
        enough to ensure 1 Lipschitzity a coertive coefficient is applied on the
        output.
        The computation is done in three steps:

        1. reduce the largest singular value to 1, using iterated power method.
        2. increase other singular values to 1, using BjorckNormalizer algorithm.
        3. divide the output by the Lipschitz bound to ensure k Lipschitzity.

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
        if padding_mode != "same":
            raise RuntimeError("NormalizedConv only support padding='same'")

        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        # self.bn = nn.BatchNorm2d(self.out_channels)
        self.niter_spectral = niter_spectral
        self.niter_bjorck = niter_bjorck
        self.k_coef_lip = k_coef_lip
        self.set_klip_factor(self.k_coef_lip)
        spectral_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
        # spectral normalization is performed during forward.
        # spectral_norm is implemented via a hook that calculates
        # spectral norm and rescales weight before every :meth:~Module.forward call.
        spectral_norm(
            self,
            name="weight",
            n_power_iterations=self.niter_spectral,
        )
        if self.niter_spectral < 1:
            raise RuntimeError("niter_spectral has to be > 0")

    def _compute_lip_coef(self, input_shape=None):
        # According to the file lipschitz_CNN.pdf
        return compute_lconv_ip_coef(self.kernel_size, input_shape, self.stride)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.is_init:
            self._init_lip_coef(input.shape[1:])
            self.is_init = True
        W_bar = (
            bjorck_normalization(self.weight, niter=self.niter_bjorck)
            * self._get_coef()
        )
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                W_bar,
                self.bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input,
            W_bar,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def condense(self):
        bjorck_normalization(self.weight, niter=self.niter_bjorck)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilatione=self.dilation,
        )
        layer.reset_parameters(self)
        setattr(self, "weight", self.weight * self._get_coef())
        if self.bias:
            setattr(self, "bias", self.bias)
        return layer

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "niter_spectral": self.niter_spectral,
            "niter_bjorck": self.niter_bjorck,
        }
        base_config = super().state_dict(destination, prefix, keep_vars)
        return dict(list(base_config.items()) + list(config.items()))


class SpectralConv3d(nn.Conv3d, LipschitzLayer, Condensable):
    is_init = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        k_coef_lip: float = 1.0,
        niter_spectral: int = DEFAULT_NITER_SPECTRAL,
        niter_bjorck: int = DEFAULT_NITER_BJORCK,
    ):
        """
        This class is a Conv3d Layer constrained such that all singular of it's kernel
        are 1. The computation based on BjorckNormalizer algorithm. As this is not
        enough to ensure 1 Lipschitzity a coertive coefficient is applied on the
        output.
        The computation is done in three steps:

        1. reduce the largest singular value to 1, using iterated power method.
        2. increase other singular values to 1, using BjorckNormalizer algorithm.
        3. divide the output by the Lipschitz bound to ensure k Lipschitzity.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to all three sides of
            the input. Default: 0
            padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel elements.
            Default: 1
            groups (int, optional): Number of blocked connections from input channels
            to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
            k_coef_lip: lipschitz constant to ensure
            niter_spectral: number of iteration to find the maximum singular value.
            niter_bjorck: number of iteration with BjorckNormalizer algorithm.


        This documentation reuse the body of the original torch.nn.Conv3d doc.
        """
        # if not ((dilation == (1, 1)) or (dilation == [1, 1]) or (dilation == 1)):
        #     raise RuntimeError("NormalizedConv does not support dilation rate")
        # if padding_mode != "same":
        #     raise RuntimeError("NormalizedConv only support padding='same'")

        nn.Conv3d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        # self.bn = nn.BatchNorm2d(self.out_channels)
        self.niter_spectral = niter_spectral
        self.niter_bjorck = niter_bjorck
        self.k_coef_lip = k_coef_lip
        self.set_klip_factor(self.k_coef_lip)
        spectral_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
        # spectral normalization is performed during forward.
        # spectral_norm is implemented via a hook that calculates
        # spectral norm and rescales weight before every :meth:~Module.forward call.
        spectral_norm(
            self,
            name="weight",
            n_power_iterations=self.niter_spectral,
        )
        if self.niter_spectral < 1:
            raise RuntimeError("niter_spectral has to be > 0")

    def _compute_lip_coef(self, input_shape=None):
        # According to the file lipschitz_CNN.pdf
        return compute_lconv_ip_coef(self.kernel_size, input_shape, self.stride)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.is_init:
            self._init_lip_coef(input.shape[1:])
            self.is_init = True
        W_bar = bjorck_normalization(self.weight, niter=self.niter_bjorck)
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                W_bar * self.lipschitz_gain,
                self.bias,
                self.stride,
                (0, 0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv3d(
            input,
            W_bar * self.lipschitz_gain,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def condense(self):
        bjorck_normalization(self.weight, niter=self.niter_bjorck)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = nn.Conv3d(
            self,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilatione=self.dilation,
        )
        layer.reset_parameters(self)
        setattr(self, "weight", self.weight * self._get_coef())
        if self.bias:
            setattr(self, "bias", self.bias)
        return layer

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "niter_spectral": self.niter_spectral,
            "niter_bjorck": self.niter_bjorck,
        }
        base_config = super().state_dict(destination, prefix, keep_vars)
        return dict(list(base_config.items()) + list(config.items()))


class FrobeniusConv2d(nn.Conv2d, LipschitzLayer, Condensable):
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
        k_coef_lip: int = 1.0,
    ):
        if not ((stride == (1, 1)) or (stride == [1, 1]) or (stride == 1)):
            raise RuntimeError("NormalizedConv does not support strides")
        # if padding_mode != "same":
        #     raise RuntimeError("NormalizedConv only support padding='same'")

        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.k_coef_lip = k_coef_lip
        self.set_klip_factor(self.k_coef_lip)

    def reset_parameters(self) -> None:
        init.orthogonal_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def _compute_lip_coef(self, input_shape=None):
        return compute_lconv_ip_coef(self.kernel_size, input_shape, self.stride)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.is_init:
            self._init_lip_coef(input.shape[1:])
            self.is_init = True
        W_bar = self.weight / np.linalg.norm(self.weight.detach().numpy())
        return self._conv_forward(input, W_bar * self._get_coef())

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super().state_dict(destination, prefix, keep_vars)
        return dict(list(base_config.items()) + list(config.items()))

    def condense(self):
        setattr(
            self, "weight", self.weight / np.linalg.norm(self.weight.detach().numpy())
        )

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        # call the condense function from SpectralDense as if it was from this class
        return SpectralConv2d.vanilla_export(self)


class ScaledAvgPool2d(nn.AvgPool2d, LipschitzLayer):
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
        # if not (stride is None):
        #     raise RuntimeError("stride must be equal to pool_size")
        # if padding != 0:
        #     raise RuntimeError("ScaledAveragePooling2D only support padding='valid'")
        nn.AvgPool2d.__init__(
            self,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        self.set_klip_factor(k_coef_lip)

    def _compute_lip_coef(self, input_shape=None):
        return math.sqrt(np.prod(np.asarray(self.kernel_size)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(input) * self._get_coef()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super().state_dict(destination, prefix, keep_vars)
        return dict(list(base_config.items()) + list(config.items()))


class ScaledGlobalAvgPool2d(nn.AdaptiveAvgPool2d, LipschitzLayer):
    def __init__(
        self,
        data_format=None,
        k_coef_lip=1.0,
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
        nn.AdaptiveAvgPool2d.__init__(1)
        self.set_klip_factor(k_coef_lip)
        self._init_lip_coef(None)

    def _compute_lip_coef(self, input_shape=None):
        lip_coef = np.sqrt(input_shape[-2] * input_shape[-1])
        return lip_coef

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            F.adaptive_avg_pool2d(
                input,
                self.output_size,
                self.return_indices,
            )
            * self._get_coef()
        )

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super().state_dict(destination, prefix, keep_vars)
        return dict(list(base_config.items()) + list(config.items()))


class InvertibleDownSampling(nn.Module):
    def __init__(
        self,
        pool_size,
    ):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                input[:, :, i :: self.pool_size[0], j :: self.pool_size[1]]
                for i in range(self.pool_size[0])
                for j in range(self.pool_size[1])
            ],
            axis=1,
        )

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        config = {
            "pool_size": self.pool_size,
        }
        base_config = super().state_dict(destination, prefix, keep_vars)
        return dict(list(base_config.items()) + list(config.items()))
