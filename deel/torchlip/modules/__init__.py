# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
from .activation import FullSort
from .activation import GroupSort
from .activation import GroupSort2
from .activation import LPReLU
from .activation import MaxMin
from .conv import FrobeniusConv2d
from .conv import SpectralConv2d,SpectralConv1d
from.normalization import LayerCentering,LayerCentering2D
from.normalization import BatchCentering, BatchCentering2D,BatchCenteringBiases,BatchCenteringBiases2D
from .downsampling import InvertibleDownSampling
from .linear import FrobeniusLinear
from .linear import SpectralLinear
from .loss import HingeMarginLoss
from .loss import HingeMulticlassLoss
from .loss import HKRLoss
from .loss import HKRMulticlassLoss,HKRMultiLoss
from .loss import KRLoss
from .loss import KRMulticlassLoss
from .loss import NegKRLoss
from .module import LipschitzModule
from .module import Sequential
from .pooling import ScaledAdaptiveAvgPool2d
from .pooling import ScaledAvgPool2d
from .pooling import ScaledL2NormPool2d
from .pooling import ScaledGlobalL2NormPooling2D,ScaledGlobalL2NormPooling
from .upsampling import InvertibleUpSampling
