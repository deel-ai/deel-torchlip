# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

from typing import Any, TypeVar

import torch

from .hook_norm import HookNorm


class FrobeniusNorm(HookNorm):
    def __init__(self, module: torch.nn.Module, name: str):
        super().__init__(module, name)

    def compute_weight(self, module: torch.nn.Module, inputs: Any) -> torch.Tensor:
        w: torch.Tensor = self.weight(module)
        return w / torch.norm(w)  # type: ignore

    @staticmethod
    def apply(module: torch.nn.Module, name: str) -> "FrobeniusNorm":
        return FrobeniusNorm(module, name)


T_module = TypeVar("T_module", bound=torch.nn.Module)


def frobenius_norm(module: T_module, name: str = "weight") -> T_module:
    r"""
    Applies Frobenius normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = \dfrac{\mathbf{w}}{\Vert{}\mathbf{w}\Vert{}}

    Args:
        module: Containing module.
        name: Name of weight parameter.

    Returns:
        The original module with the Frobenius normalization hook.

    Example::

        >>> m = frobenius_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)

    """
    FrobeniusNorm.apply(module, name)
    return module


def remove_frobenius_norm(module: T_module, name: str = "weight") -> T_module:
    r"""
    Removes the Frobenius normalization hook from a module.

    Args:
        module: Containing module.
        name: Name of weight parameter.

    Example:
        >>> m = frobenius_norm(nn.Linear(20, 40))
        >>> remove_frobenius_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, FrobeniusNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("frobenius_norm of '{}' not found in {}".format(name, module))
