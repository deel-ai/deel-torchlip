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
# flake8: noqa

from os import path

with open(path.join(path.dirname(__file__), "VERSION")) as f:
    __version__ = f.read().strip()

from . import functional, init, normalizers, utils
from .modules import *

__all__ = [
    "FrobeniusConv2d",
    "FrobeniusLinear",
    "FullSort",
    "GroupSort",
    "GroupSort2",
    "HKRLoss",
    "HKRMulticlassLoss",
    "SoftHKRMulticlassLoss",
    "LseHKRMulticlassLoss",
    "HingeMarginLoss",
    "HingeMulticlassLoss",
    "InvertibleDownSampling",
    "InvertibleUpSampling",
    "KRLoss",
    "KRMulticlassLoss",
    "LipschitzModule",
    "LipschitzPReLU",
    "MaxMin",
    "NegKRLoss",
    "ScaledAdaptiveAvgPool2d",
    "ScaledAvgPool2d",
    "ScaledL2NormPooling2D",
    "Sequential",
    "SpectralConv2d",
    "SpectralLinear",
    "vanilla_model",
]
