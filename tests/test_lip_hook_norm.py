# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

import torch
import torch_testing as tt

from deel.torchlip.utils import (
    bjorck_norm,
    remove_bjorck_norm,
    frobenius_norm,
    remove_frobenius_norm,
    lconv_norm,
    remove_lconv_norm,
)
from deel.torchlip.utils.lconv_norm import compute_lconv_coef
from deel.torchlip.normalizers import bjorck_normalization


def test_bjorck_norm():
    """
    test bjorck_norm hook implementation
    """
    m = torch.nn.Linear(2, 2)
    torch.nn.init.orthogonal_(m.weight)
    w1 = bjorck_normalization(m.weight)

    bjorck_norm(m)
    assert not isinstance(m.weight, torch.nn.Parameter)

    x = torch.rand(2)
    m(x)
    tt.assert_equal(w1, m.weight)

    remove_bjorck_norm(m)
    assert isinstance(m.weight, torch.nn.Parameter)
    tt.assert_equal(w1, m.weight)


def test_frobenius_norm():
    """
    test frobenius_norm hook implementation
    """
    m = torch.nn.Linear(2, 2)
    torch.nn.init.uniform_(m.weight)
    w1 = m.weight / torch.norm(m.weight)

    frobenius_norm(m)
    assert not isinstance(m.weight, torch.nn.Parameter)

    x = torch.rand(2)
    m(x)
    tt.assert_equal(w1, m.weight)

    remove_frobenius_norm(m)
    assert isinstance(m.weight, torch.nn.Parameter)
    tt.assert_equal(w1, m.weight)


def test_lconv_norm():
    """
    test lconv_norm hook implementation
    """
    m = torch.nn.Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
    torch.nn.init.orthogonal_(m.weight)
    w1 = m.weight * compute_lconv_coef(m.kernel_size, (1, 1, 5, 5), m.stride)

    lconv_norm(m)
    assert not isinstance(m.weight, torch.nn.Parameter)

    x = torch.rand(1, 1, 5, 5)
    m(x)
    tt.assert_equal(w1, m.weight)

    remove_lconv_norm(m)
    assert isinstance(m.weight, torch.nn.Parameter)
    tt.assert_equal(w1, m.weight)
