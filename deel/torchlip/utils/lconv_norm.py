# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================


from typing import Any, Tuple

import numpy as np
import torch

from .hook_norm import HookNorm


def compute_lconv_coef(
    kernel_size: Tuple[int, ...],
    input_shape: Tuple[int, ...],
    strides: Tuple[int, ...] = (1, 1),
) -> float:
    # See https://arxiv.org/abs/2006.06520
    stride = np.prod(strides)
    k1, k2 = kernel_size
    h, w = input_shape[-2:]

    k1_div2 = (k1 - 1) / 2
    k2_div2 = (k2 - 1) / 2

    if stride == 1:
        coefLip = np.sqrt(
            (w * h)
            / ((k1 * h - k1_div2 * (k1_div2 + 1)) * (k2 * w - k2_div2 * (k2_div2 + 1)))
        )
    else:
        sn1 = strides[0]
        sn2 = strides[1]
        ho = np.floor(h / sn1)
        wo = np.floor(w / sn2)
        alphabar1 = np.floor(k1_div2 / sn1)
        alphabar2 = np.floor(k2_div2 / sn2)
        betabar1 = k1_div2 - alphabar1 * sn1
        betabar2 = k2_div2 - alphabar2 * sn2
        zl1 = (alphabar1 * sn1 + 2 * betabar1) * (alphabar1 + 1) / 2
        zl2 = (alphabar2 * sn2 + 2 * betabar2) * (alphabar2 + 1) / 2
        gamma1 = h - 1 - sn1 * np.ceil((h - 1 - k1_div2) / sn1)
        gamma2 = w - 1 - sn2 * np.ceil((w - 1 - k2_div2) / sn2)
        alphah1 = np.floor(gamma1 / sn1)
        alphaw2 = np.floor(gamma2 / sn2)
        zr1 = (alphah1 + 1) * (k1_div2 - gamma1 + sn1 * alphah1 / 2.0)
        zr2 = (alphaw2 + 1) * (k2_div2 - gamma2 + sn2 * alphaw2 / 2.0)
        coefLip = np.sqrt((h * w) / ((k1 * ho - zl1 - zr1) * (k2 * wo - zl2 - zr2)))

    return coefLip  # type: ignore


class LConvNorm(HookNorm):

    """
    Kernel normalization for Lipschitz convolution. Normalize weights
    based on input shape and kernel size, see https://arxiv.org/abs/2006.06520
    """

    @staticmethod
    def apply(module: torch.nn.Module) -> "LConvNorm":

        if not isinstance(module, torch.nn.Conv2d):
            raise RuntimeError(
                "Can only apply lconv_norm hooks on 2D-convolutional layer."
            )

        return LConvNorm(module, "weight")

    def compute_weight(self, module: torch.nn.Module, inputs: Any) -> torch.Tensor:
        assert isinstance(module, torch.nn.Conv2d)
        coefficient = compute_lconv_coef(
            module.kernel_size, inputs[0].shape[-4:], module.stride
        )
        return self.weight(module) * coefficient


def lconv_norm(module: torch.nn.Conv2d) -> torch.nn.Conv2d:
    r"""
    Applies Lipschitz normalization to a kernel in the given convolutional.

    Args:
        module: Containing module.

    Returns:
        The original module with the Frobenius normalization hook.

    Example::

        >>> m = lconv_norm(nn.Conv2d(16, 16, (3, 3)))
        >>> m
        Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))

    """
    LConvNorm.apply(module)
    return module


def remove_lconv_norm(module: torch.nn.Conv2d) -> torch.nn.Conv2d:
    r"""
    Removes the Lipschitz normalization hook from a module.

    Args:
        module: Containing module.

    Example:

        >>> m = lconv_norm(nn.Conv2d(16, 16, (3, 3)))
        >>> remove_lconv_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, LConvNorm):
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("lconv_norm not found in {}".format(module))
