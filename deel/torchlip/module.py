# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains equivalents for Model and Sequential. These classes add support
for condensation and vanilla exportation.
"""

import copy
import logging
import math

from typing import Any

from collections import OrderedDict
import numpy as np
from torch.nn import Sequential as TorchSequential

from .layers import LipschitzModule

logger = logging.getLogger("deel.torchlip")


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
        LipschitzModule.__init__(self, k_coef_lip, 1)

        # Force the Lipschitz coefficient:
        n_layers = np.sum(
            (isinstance(layer, LipschitzModule) for layer in self.children())
        )
        for module in self.children():
            if isinstance(module, LipschitzModule):
                module._coefficient_lip = math.pow(k_coef_lip, 1 / n_layers)
            else:
                logger.warning(
                    "Sequential model contains a layer wich is not a Lipschitsz layer: {}".format(  # noqa: E501
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
        for layer in self.children():
            if isinstance(layer, LipschitzModule):
                layers.append(layer.vanilla_export())
            else:
                layers.append(copy.deepcopy(layer))
        return TorchSequential(*layers)
