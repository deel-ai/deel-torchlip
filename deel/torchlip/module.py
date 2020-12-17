# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains equivalents for Model and Sequential. These classes add support
for condensation and vanilla exportation.
"""
import math
from warnings import warn
from collections import OrderedDict
import numpy as np
from torch.nn import Sequential as TorchSequential

from .layers import LipschitzModule


class Sequential(TorchSequential, LipschitzModule):
    def __init__(
        self,
        layers=None,
        k_coef_lip=1.0,
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
        super(Sequential, self).__init__(layers)
        self.layers = layers
        if len(self.layers) == 1 and isinstance(self.layers[0], OrderedDict):
            self.layers_list = self.layers[0].values()
        else:
            self.layers_list = self.layers
        self.set_klip_factor(k_coef_lip)
        self.init = False

    def forward(self, input):
        if not self.init:
            self._init_lip_coef(input.shape[1:])
            self.init = True
        return super(Sequential, self).forward(input)

    def set_klip_factor(self, klip_factor):
        super(Sequential, self).set_klip_factor(klip_factor)
        nb_layers = np.sum([isinstance(layer, LipschitzModule) for layer in self.layers])
        for module in enumerate(self.layers_list):
            if isinstance(module, LipschitzModule):
                module.set_klip_factor(math.pow(klip_factor, 1 / nb_layers))
            else:
                warn(
                    "Sequential model contains a layer wich is not a Lipschitsz layer: {}".format(  # noqa: E501
                        module
                    )
                )

    def _compute_lip_coef(self, input_shape=None):
        for layer in self.layers_list:
            if isinstance(layer, LipschitzModule):
                layer._compute_lip_coef(input_shape)
            else:
                warn(
                    "Sequential module contains a layer wich is not a Lipschitsz layer: {}".format(  # noqa: E501
                        layer
                    )
                )

    def _init_lip_coef(self, input_shape):
        for layer in self.layers_list:
            if isinstance(layer, LipschitzModule):
                layer._init_lip_coef(input_shape)
            else:
                warn(
                    "Sequential model contains a layer wich is not a Lipschitsz layer: {}".format(  # noqa: E501
                        layer
                    )
                )

    def _get_coef(self):
        global_coef = 1.0
        for layer in self.layers_list:
            if isinstance(layer, LipschitzModule) and (global_coef is not None):
                global_coef *= layer._get_coef()
            else:
                warn(
                    "Sequential model contains a layer wich is not a Lipschitsz layer: {}".format(  # noqa: E501
                        layer
                    )
                )
                global_coef = None
        return global_coef

    def condense(self):
        for layer in self.layers_list:
            if isinstance(layer):
                layer.condense()

    def vanilla_export(self):
        layers = list()
        for layer in self.layers_list:
            if isinstance(layer):
                layer.condense()
                layers.append(layer.vanilla_export())
            else:
                lay_cp = layer.__class__.from_config(layer.get_config())
                lay_cp.build(layer.input.shape[1:])
                lay_cp.set_weights(layer.get_weights())
                layers.append(lay_cp)
        model = TorchSequential(layers, self.name)
        return model
