# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains extra activation functions which respect the Lipschitz constant.
It can be added as a layer, or it can be used in the "activation" params for other
layers.
"""
from torch import nn, Tensor, cat, split, minimum, maximum, sort
from .layers import TorchLipschitzLayer
from .utils import _deel_export


@_deel_export
class MaxMin(nn.Module, TorchLipschitzLayer):
    def __init__(self, data_format="channels_last", k_coef_lip=1.0, *args, **kwargs):
        """
        MaxMin activation [Relu(x),reLU(-x)]

        Args:
            data_format: either channels_first or channels_last
            k_coef_lip: the lipschitz coefficient to be enforced
            *args: params passed to Layers
            **kwargs: params passed to layers (named fashion)

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does
            not include the samples axis) when using this layer as the first layer in a
            model.

        Output shape:
            Double channel size as input.

        References:
            ([M. Blot, M. Cord, et N. Thome, « Max-min convolutional neural networks
            for image classification », in 2016 IEEE International Conference on Image
            Processing (ICIP), Phoenix, AZ, USA, 2016, p. 3678‑3682.)

        """
        self.set_klip_factor(k_coef_lip)
        super(MaxMin, self).__init__(*args, **kwargs)
        self.init = False

    # def reset_parameters(self) -> None:
    #     return super().reset_parameters()

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    def forward(self, input: Tensor) -> Tensor:
        if not self.init:
            self._init_lip_coef(input.shape[1:])
            self.init = True
        return (
            cat((nn.functional.relu(input), nn.functional.relu(-input)), 1)
            * self._get_coef()
        )

    def state_dict(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(MaxMin, self).state_dict()
        return dict(list(base_config.items()) + list(config.items()))


@_deel_export
class GroupSort(nn.Module, TorchLipschitzLayer):
    def __init__(self, n=None, k_coef_lip=1.0, *args, **kwargs):
        """
        GroupSort activation

        Args:
            n: group size used when sorting. When None group size is set to input
                size (fullSort behavior)
            data_format: either channels_first or channels_last
            k_coef_lip: the lipschitz coefficient to be enforced
            *args: params passed to Layers
            **kwargs: params passed to layers (named fashion)

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does
            not include the samples axis) when using this layer as the first layer in a
            model.

        Output shape:
            Same size as input.

        """
        self.set_klip_factor(k_coef_lip)
        super(GroupSort, self).__init__(*args, **kwargs)
        self.n = n
        self.init = False

    # def reset_parameters(self) -> None:
    #     super(GroupSort, self).reset_parameters()
    #     self._init_lip_coef(input_shape)
    #     if (self.n is None) or (self.n > input_shape[self.channel_axis]):
    #         self.n = input_shape[self.channel_axis]
    #     if (input_shape[self.channel_axis] % self.n) != 0:
    #         raise RuntimeError("self.n has to be a divisor of the number of channels")
    #     print(self.n)

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    def forward(self, input: Tensor) -> Tensor:
        if not self.init:
            self._init_lip_coef(input.shape[1:])
            self.init = True
        fv = input.reshape([-1, self.n])
        if self.n == 2:
            b, c = split(fv, 2, 1)
            newv = cat([minimum(b, c), maximum(b, c)], axis=1)
            newv = newv.reshape(input.shape)
            return newv

        newv = sort(fv)
        newv = newv.reshape(newv, input.shape)
        return newv * self._get_coef()

    def state_dict(self):
        config = {
            "n": self.n,
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(GroupSort, self).state_dict()
        return dict(list(base_config.items()) + list(config.items()))


@_deel_export
class GroupSort2(GroupSort):
    def __init__(self, **kwargs):
        """
        GroupSort2 activation. Special case of GroupSort with group of size 2.

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does
            not include the samples axis) when using this layer as the first layer in a
            model.

        Output shape:
            Same size as input.

        """
        kwargs["n"] = 2
        super().__init__(**kwargs)


@_deel_export
class FullSort(GroupSort):
    def __init__(self, **kwargs):
        """
        FullSort activation. Special case of GroupSort where the entire input is sorted.

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does
            not include the samples axis) when using this layer as the first layer in a
            model.

        Output shape:
            Same size as input.

        """
        kwargs["n"] = None
        super().__init__(**kwargs)


@_deel_export
def PReLUlip(input, weight, k_coef_lip=1.0):
    """
    PreLu activation, with Lipschitz constraint.

    Args:
        k_coef_lip: lipschitz coefficient to be enforced
    """
    return nn.functional.prelu(input, weight) * k_coef_lip
