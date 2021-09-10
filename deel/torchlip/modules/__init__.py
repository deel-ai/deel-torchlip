# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module extends original Pytorch layers, in order to add k-Lipschitz constraint via
re-parametrization. Currently, are implemented:

* Linear layer:
    as SpectralLinear
* Conv1d layer:
    as SpectralConv1d
* Conv2d layer:
    as SpectralConv2d
* Conv3d layer:
    as SpectralConv3d
* AvgPool2d:
    as ScaledAvgPool2d

By default the layers are 1 Lipschitz almost everywhere, which is efficient for
wasserstein distance estimation. However for other problems (such as adversarial
robustness) the user may want to use layers that are at most 1 lipschitz, this can
be done by setting the param `niter_bjorck=0`.
"""

# flake8: noqa

from .module import LipschitzModule, Sequential
from .linear import SpectralLinear, FrobeniusLinear
from .conv import SpectralConv2d, FrobeniusConv2d
from .activation import MaxMin, GroupSort, GroupSort2, FullSort, LPReLU
from .loss import (
    KRLoss,
    NegKRLoss,
    HingeMarginLoss,
    HKRLoss,
    KRMulticlassLoss,
    HingeMulticlassLoss,
    HKRMulticlassLoss,
)
from .pooling import ScaledAvgPool2d, ScaledAdaptiveAvgPool2d, ScaledL2NormPool2d
from .downsampling import InvertibleDownSampling
from .upsampling import InvertibleUpSampling
