# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

from typing import Tuple, Union

import numpy as np
import torch

from .. import functional as F
from .module import LipschitzModule


class InvertibleUpSampling(torch.nn.Module, LipschitzModule):
    def __init__(
        self, kernel_size: Union[int, Tuple[int, ...]], k_coef_lip: float = 1.0
    ):
        torch.nn.Module.__init__(self)
        LipschitzModule.__init__(self, k_coef_lip)
        self.kernel_size = kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.invertible_upsample(input, self.kernel_size) * self._coefficient_lip
