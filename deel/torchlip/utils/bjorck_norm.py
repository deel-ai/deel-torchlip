# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

from typing import Any, TypeVar

import torch

from .hook_norm import HookNorm
from ..normalizers import DEFAULT_NITER_BJORCK, bjorck_normalization


class BjorckNorm(HookNorm):
    """
    Bjorck Normalization from https://arxiv.org/abs/1811.05381
    """

    n_iterations: int

    def __init__(self, module: torch.nn.Module, name: str, n_iterations: int):
        super().__init__(module, name)
        self.n_iterations = n_iterations

    def compute_weight(self, module: torch.nn.Module, inputs: Any) -> torch.Tensor:
        return bjorck_normalization(self.weight(module), self.n_iterations)

    @staticmethod
    def apply(module: torch.nn.Module, name: str, n_iterations: int) -> "BjorckNorm":
        return BjorckNorm(module, name, n_iterations)


T_module = TypeVar("T_module", bound=torch.nn.Module)


def bjorck_norm(
    module: T_module, name: str = "weight", n_iterations: int = DEFAULT_NITER_BJORCK
) -> T_module:
    r"""
    Applies Bjorck normalization to a parameter in the given module.

    Bjorck normalization ensures that all eigen values of a vectors remain close or
    equal to one during training. If the dimension of the weight tensor is greater than
    2, it is reshaped to 2D for iteration.
    This is implemented via a hook that applies Bjorck normalization before every
    ``forward()`` call.

    .. note::
        It is recommended to use :py:func:`torch.nn.utils.spectral_norm` before
        this hook to greatly reduce the number of iterations required.

    See `Sorting out Lipschitz function approximation <https://arxiv.org/abs/1811.05381>`_.

    Args:
        module: Containing module.
        name: Name of weight parameter.
        n_iterations: Number of iterations for the normalization.

    Returns:
        The original module with the Bjorck normalization hook.

    Example::

        >>> m = bjorck_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)

    """
    BjorckNorm.apply(module, name, n_iterations)
    return module


def remove_bjorck_norm(module: T_module, name: str = "weight") -> T_module:
    r"""
    Removes the Bjorck normalization reparameterization from a module.

    Args:
        module: Containing module.
        name: Name of weight parameter.

    Example:
        >>> m = bjorck_norm(nn.Linear(20, 40))
        >>> remove_bjorck_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BjorckNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("bjorck_norm of '{}' not found in {}".format(name, module))
