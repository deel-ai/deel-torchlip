# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

from abc import abstractmethod
from typing import Any

import inflection
import torch


class HookNorm:

    """
    Base class for pre-forward hook that modifies parameters of a module. The
    constructor register the hook on the module, and sub-classes should only
    implement the compute_weight method.
    """

    _name: str
    _first: bool

    def __init__(self, module: torch.nn.Module, name: str = "weight"):
        self._name = name
        self._first = False

        if isinstance(getattr(module, name), torch.nn.Parameter):
            weight = module._parameters[name]
            self._first = True
            delattr(module, name)
            module.register_parameter(name + "_orig", weight)
            setattr(module, name, weight.data)

        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, type(self)) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two {} hooks on "
                    "the same parameter {}.".format(
                        inflection.underscore(type(self).__name__), name
                    )
                )

        # Normalize weight before every forward().
        module.register_forward_pre_hook(self)

    def weight(self, module: torch.nn.Module) -> torch.Tensor:
        """
        Returns:
            The weight to apply the transformation to. This is not always the value
            of the attribute corresponding to `name`.
        """
        if self._first:
            weight = getattr(module, self._name + "_orig")
        else:
            weight = getattr(module, self._name)
        return weight  # type: ignore

    @property
    def name(self) -> str:
        """
        Returns:
            The name of the attribute that should be set by this hook.
        """
        return self._name

    def remove(self, module: torch.nn.Module):
        # If this was the first layer to hook, we reset the weights.
        if self._first:
            weight = getattr(module, self._name)
            delattr(module, self._name)
            module.register_parameter(self._name, torch.nn.Parameter(weight.detach()))

    @abstractmethod
    def compute_weight(self, module: torch.nn.Module, inputs: Any) -> torch.Tensor:
        """
        Transform the weight of the given module.
        """
        pass

    def __call__(self, module: torch.nn.Conv2d, inputs: Any):
        setattr(module, self.name, self.compute_weight(module, inputs))
