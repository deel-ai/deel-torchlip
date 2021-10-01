# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

# flake8: noqa

__version__ = "0.0.1"

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
]
