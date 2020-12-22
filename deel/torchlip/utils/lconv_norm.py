r"""
Kernel normalization for Lipschitz convolution. Normalize weights
based on input shape and kernel size, see https://arxiv.org/abs/2006.06520
"""

from typing import Any, Tuple

import numpy as np
import torch


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


class LConvNorm:
    @staticmethod
    def apply(module) -> "LConvNorm":
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, LConvNorm):
                raise RuntimeError(
                    "Cannot register two lconv_norm hooks on the same module"
                )

        if not isinstance(module, torch.nn.Conv2d):
            raise RuntimeError(
                "Can only apply lconv_norm hooks on 2D-convolutional layer."
            )

        fn = LConvNorm()

        # Normalize weight before every forward().
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: torch.nn.Module):
        # Nothing to do here, just keeping the method to be
        # consistent with torch hook classes.
        pass

    def __call__(self, module: torch.nn.Conv2d, inputs: Any):
        weight = module.weight
        kernel_size = module.kernel_size
        strides = module.stride

        setattr(
            module,
            "weight",
            weight * compute_lconv_coef(kernel_size, inputs[0].shape[-4:], strides),
        )


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
