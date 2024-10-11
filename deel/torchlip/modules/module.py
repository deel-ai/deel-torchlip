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
"""
This module contains equivalents for Model and Sequential. These classes add support
for condensation and vanilla exportation.
"""
import abc
import copy
import logging
import warnings
import math
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn import Sequential as TorchSequential
from torch import reshape


def _is_supported_1lip_layer(layer):
    """Return True if the Keras layer is 1-Lipschitz. Note that in some cases, the layer
    is 1-Lipschitz for specific set of parameters.
    """
    supported_1lip_layers = (
        torch.nn.Softmax,
        torch.nn.Flatten,
        torch.nn.Identity,
        torch.nn.ReLU,
        torch.nn.Sigmoid,
        torch.nn.Tanh,
        Reshape,
    )
    if isinstance(layer, supported_1lip_layers):
        return True
    elif isinstance(layer, torch.nn.MaxPool2d):
        return layer.kernel_size <= layer.stride
    return False


def vanilla_model(model: nn.Module):
    """Convert lipschitz modules into their non-lipschitz counterpart (for
    instance, SpectralConv2d layers become Conv2d layers).

    Warning: This function modifies the model in-place.

    Args:
        model (nn.Module): Lipschitz neural network
    """
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            # compound module, go inside it
            vanilla_model(module)

        if isinstance(module, LipschitzModule):
            # simple module
            setattr(model, n, module.vanilla_export())


class _LipschitzCoefMultiplication(nn.Module):
    """Parametrization module for lipschitz global coefficient multiplication."""

    def __init__(self, coef: float):
        super().__init__()
        self._coef = coef

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return self._coef * weight


class LipschitzModule(abc.ABC):
    """
    This class allow to set lipschitz factor of a layer. Lipschitz layer must inherit
    this class to allow user to set the lipschitz factor.

    Warning:
         This class only regroup useful functions when developing new Lipschitz layers.
         But it does not ensure any property about the layer. This means that
         inheriting from this class won't ensure anything about the lipschitz constant.
    """

    # The target coefficient:
    _coefficient_lip: float

    def __init__(self, coefficient_lip: float = 1.0):
        self._coefficient_lip = coefficient_lip

    def apply_lipschitz_factor(self):
        """Multiply the layer weights by a lipschitz factor."""
        if self._coefficient_lip == 1.0:
            return
        parametrize.register_parametrization(
            self, "weight", _LipschitzCoefMultiplication(self._coefficient_lip)
        )

    @abc.abstractmethod
    def vanilla_export(self):
        """
        Convert this layer to a corresponding vanilla torch layer (when possible).

        Returns:
             A vanilla torch version of this layer.
        """
        pass


class Sequential(TorchSequential, LipschitzModule):
    def __init__(
        self,
        *args: Any,
        k_coef_lip: float = 1.0,
    ):
        """
        Equivalent of torch.Sequential but allow to set k-lip factor globally. Also
        support condensation and vanilla exportation.
        For now constant repartition is implemented (each layer
        get n_sqrt(k_lip_factor), where n is the number of layers)
        But in the future other repartition function may be implemented.

        Args:
            layers: list of layers to add to the model.
            name: name of the model, can be None
            k_coef_lip: the Lipschitz coefficient to ensure globally on the model.
        """
        TorchSequential.__init__(self, *args)
        LipschitzModule.__init__(self, k_coef_lip)

        # Force the Lipschitz coefficient:
        n_layers = np.sum(
            [isinstance(layer, LipschitzModule) for layer in self.children()]
        )
        for module in self.children():
            if isinstance(module, LipschitzModule):
                module._coefficient_lip = math.pow(k_coef_lip, 1 / n_layers)
            elif _is_supported_1lip_layer(module) is not True:
                warnings.warn(
                    "Sequential model contains a layer which is not a Lipschitz layer: {}".format(  # noqa: E501
                        module
                    )
                )

    def vanilla_export(self):
        """
        Exports this model to a vanilla torch Sequential.

        This method only works for flat sequential. Lipschitz modules are converted
        using their own `vanilla_export` method while non-Lipschitz modules are simply
        copied using `copy.deepcopy`.

        Returns:
            A Vanilla torch.nn.Sequential model.
        """
        layers = []
        for name, layer in self.named_children():
            if isinstance(layer, LipschitzModule):
                layers.append((name, layer.vanilla_export()))
            else:
                layers.append((name, copy.deepcopy(layer)))
        return TorchSequential(OrderedDict(layers))


class Reshape(torch.nn.Module):
    def __init__(self, target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return reshape(x, self.target_shape)
