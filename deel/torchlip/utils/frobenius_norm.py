r"""
Frobenius Normalization.
"""

from typing import Any, TypeVar

import torch


class FrobeniusNorm:
    name: str
    first: bool

    def __init__(self, name: str):
        self.name = name
        self.first = False

    def compute_weight(self, module: torch.nn.Module) -> torch.Tensor:
        w: torch.Tensor
        if self.first:
            w = getattr(module, self.name + "_orig")
        else:
            w = getattr(module, self.name)
        return w / torch.norm(w)  # type: ignore

    @staticmethod
    def apply(module, name: str) -> "FrobeniusNorm":
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, FrobeniusNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two frobenius_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = FrobeniusNorm(name)

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
