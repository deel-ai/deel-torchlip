r"""
Bjorck Normalization from https://arxiv.org/abs/1811.05381
"""

from typing import Any, TypeVar

import torch

from ..normalizers import DEFAULT_NITER_BJORCK, bjorck_normalization


class BjorckNorm:
    name: str
    n_iterations: int
    first: bool

    def __init__(self, name: str, n_iterations: int):
        self.name = name
        self.n_iterations = n_iterations
        self.first = False

    def compute_weight(self, module: torch.nn.Module) -> torch.Tensor:
        w: torch.Tensor
        if self.first:
            w = getattr(module, self.name + "_orig")
        else:
            w = getattr(module, self.name)
        return bjorck_normalization(w, self.n_iterations)  # type: ignore

    @staticmethod
    def apply(module, name: str, n_iterations: int) -> "BjorckNorm":
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, BjorckNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two bjorck_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = BjorckNorm(name, n_iterations)

        if isinstance(getattr(module, name), torch.nn.Parameter):
            weight = module._parameters[name]
            fn.first = True
            delattr(module, fn.name)
            module.register_parameter(fn.name + "_orig", weight)
            setattr(module, fn.name, weight.data)

        # Normalize weight before every forward().
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: torch.nn.Module):
        # Nothing to do here, just keeping the method to be
        # consistent with torch hook classes.
        pass

    def __call__(self, module: torch.nn.Module, inputs: Any):
        setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar("T_module", bound=torch.nn.Module)


def bjorck_norm(
    module: T_module, name: str = "weight", n_iterations: int = DEFAULT_NITER_BJORCK
) -> T_module:
    r"""
    Applies Bjorck normalization to a parameter in the given module.

    See https://arxiv.org/abs/1811.05381

    Args:
        module: Containing module.
        name: Name of weight parameter.
        n_iterations: Number of iterations for the normalizaiton.

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
    Removes the Bjorck normalization hook from a module.

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
