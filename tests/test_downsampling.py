# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

import torch
import torch_testing as tt

from deel.torchlip.functional import invertible_downsample


def test_invertible_downsample():

    # 1D input
    x = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])
    x = invertible_downsample(x, (2,))
    assert x.shape == (1, 4, 2)

    # TODO: Check this.
    tt.assert_equal(x, torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8]]]))

    # 2D input
    x = torch.rand(10, 1, 128, 128)
    assert invertible_downsample(x, (4, 4)).shape == (10, 16, 32, 32)

    x = torch.rand(10, 4, 64, 64)
    assert invertible_downsample(x, (2, 2)).shape == (10, 16, 32, 32)

    # 3D input
    x = torch.rand(10, 2, 128, 64, 64)
    assert invertible_downsample(x, 2).shape == (10, 16, 64, 32, 32)