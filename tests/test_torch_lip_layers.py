# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import math
import os
import pprint
import unittest

import numpy as np
import torch.autograd as autograd
from deel.torchlip.layers import (
    FrobeniusConv2d,
    FrobeniusLinear,
    ScaledAvgPool2d,
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
    SpectralLinear,
    LipschitzLayer,
)
from deel.torchlip.module import Sequential
from deel.torchlip.utils import evaluate_lip_const
from torch import Tensor, abs, from_numpy, load, manual_seed, no_grad, save
from torch.nn import Conv1d, Conv2d, Conv3d, Linear, MSELoss
from torch.nn import Sequential as tSequential
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

pp = pprint.PrettyPrinter(indent=4)

"""
About these tests:
==================

What is tested:
---------------
- layer instantiation
- training
- prediction
- storing on disk and reloading
- k lip_constraint is respected ( at +-0.001 )

What is not tested:
-------------------
- layer performance ( time / accuracy )
- layer structure ( don't check that SpectralConv2D is actually a convolution )

However, all run generate log that can be manually checked with tensorboard
"""


def linear_generator(batch_size, input_shape: tuple, kernel):
    """
    Generate data according to a linear kernel
    Args:
        batch_size: size of each batch
        input_shape: shape of the desired input
        kernel: kernel used to generate data, must match the last dimensions of
            `input_shape`

    Returns:
        a generator for the data

    """
    # input_shape = tuple(input_shape)
    while True:
        # pick random sample in [0, 1] with the input shape
        batch_x = np.array(
            np.random.uniform(-10, 10, (batch_size,) + input_shape), dtype=np.float16
        )

        # apply the k lip linear transformation
        batch_y = np.tensordot(
            batch_x,
            kernel,
            axes=(
                [i for i in range(1, len(input_shape) + 1)],
                [i for i in range(0, len(input_shape))],
            ),
        )
        return (from_numpy(batch_x), from_numpy(batch_y))


def build_kernel(in_features: tuple, out_features: tuple, k=1.0):
    """
    build a kernel with defined lipschitz factor

    Args:
        in_features: input shape of the linear function
        out_features: output shape of the linear function
        k: lipshitz factor of the function

    Returns:
        the kernel for use in the linear_generator

    """
    input_shape = tuple(in_features)
    output_shape = tuple(out_features)
    kernel = np.array(
        np.random.random_sample(input_shape + output_shape), dtype=np.float16
    )
    kernel = (
        kernel * k / np.linalg.norm(kernel)
    )  # assuming lipschitz constraint is independent with respect to the chosen metric

    return kernel


def generate_k_lip_model(layer_type: type, layer_params: dict, k):
    """
    build a model with a single layer of given type, with defined lipshitz factor.

    Args:
        layer_type: the type of layer to use
        layer_params: parameter passed to constructor of layer_type
        input_shape: the shape of the input
        k: lipshitz factor of the function

    Returns:
        a torch Model with a single layer.

    """
    if issubclass(layer_type, Sequential):
        model = layer_type(layer_params)
        model.set_klip_factor(k)
        return model
    if issubclass(layer_type, LipschitzLayer):
        layer_params["k_coef_lip"] = k

    layer = layer_type(**layer_params)
    assert isinstance(layer, LipschitzLayer) or isinstance(layer, Linear)
    return tSequential(layer)


def train(train_dl, model, loss_fn, optimizer, epoch, batch_size):
    for epoch in range(epoch):
        model.train()

        for xb, yb in train_dl:
            pred = model(xb.float())
            loss = loss_fn(pred, yb.float())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(model, train_dl, loss_fn):
    model.eval()
    test_loss = 0
    runnning_mae = 0
    runnning_mse = 0
    with no_grad():
        for x, y in train_dl:
            output = model(x.float())
            test_loss += loss_fn(output, y.float())  # sum up batch loss
            error = abs(output - y).sum().data
            squared_error = ((output - y) * (output - y)).sum().data
            runnning_mae += error
            runnning_mse += squared_error

    test_loss /= len(train_dl.dataset)
    mse = math.sqrt(runnning_mse / len(train_dl.dataset))
    return test_loss, mse


def compute_output_shape(in_size, mod):
    """
    Compute output size of Module `mod` given an input with size `in_size`.
    """

    f = mod.forward(autograd.Variable(Tensor(1, *in_size)))
    return tuple(f.shape[1:])


def train_k_lip_model(
    layer_type: type,
    layer_params: dict,
    batch_size: int,
    steps_per_epoch: int,
    epochs: int,
    input_shape: tuple,
    in_features: int,
    out_features: int,
    k_lip_model: float,
    k_lip_data: float,
):
    """
    Create a generator, create a model, train it and return the results.

    Args:
        layer_type:
        layer_params:
        batch_size:
        steps_per_epoch:
        epochs:
        input_shape:
        k_lip_model:
        k_lip_data:
        **kwargs:

    Returns:
        the generator

    """
    np.random.seed(42)
    # create the keras model, defin opt, and compile it
    model = generate_k_lip_model(layer_type, layer_params, k_lip_model)
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = MSELoss(reduction="sum")
    # create the synthetic data generator
    # define logging features
    logdir = os.path.join("logs", "torch_lip_layers", "%s" % layer_type.__name__)
    os.makedirs(logdir, exist_ok=True)
    if issubclass(layer_type, (Conv1d, Conv2d, Conv3d)) or issubclass(
        layer_type, Sequential
    ):
        output_shape = compute_output_shape(input_shape, model)
        kernel = build_kernel(input_shape, output_shape, k_lip_data)
    else:
        kernel = build_kernel((in_features,), (out_features,), k_lip_data)

    if issubclass(layer_type, (Conv1d, Conv2d, Conv3d)) or issubclass(
        layer_type, Sequential
    ):
        (batch_x, batch_y) = linear_generator(batch_size, input_shape, kernel)
    else:
        (batch_x, batch_y) = linear_generator(batch_size, (in_features,), kernel)

    train_ds = TensorDataset(batch_x, batch_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    train(
        train_dl,
        model,
        loss_fn,
        optimizer,
        epochs,
        batch_size,
    )
    # the seed is set to compare all models with the same data
    if issubclass(layer_type, (Conv1d, Conv2d, Conv3d)) or issubclass(
        layer_type, Sequential
    ):
        (x, y) = linear_generator(batch_size, input_shape, kernel)
    else:
        (x, y) = linear_generator(batch_size, (in_features,), kernel)

    np.random.seed(42)
    manual_seed(42)
    if issubclass(layer_type, (Conv1d, Conv2d, Conv3d)) or issubclass(
        layer_type, Sequential
    ):
        (batch_x, batch_y) = linear_generator(batch_size, input_shape, kernel)
    else:
        (batch_x, batch_y) = linear_generator(batch_size, (in_features,), kernel)
    train_ds = TensorDataset(batch_x, batch_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    loss, mse = test(model, train_dl, loss_fn)
    empirical_lip_const = evaluate_lip_const(model=model, x=x, seed=42)
    # save the model
    model_checkpoint_path = os.path.join(logdir, "model.h5")
    save(model, model_checkpoint_path)

    del model

    model = load(model_checkpoint_path)
    model.eval()
    np.random.seed(42)
    manual_seed(42)
    if issubclass(layer_type, (Conv1d, Conv2d, Conv3d)) or issubclass(
        layer_type, Sequential
    ):
        (batch_x, batch_y) = linear_generator(batch_size, input_shape, kernel)
    else:
        (batch_x, batch_y) = linear_generator(batch_size, (in_features,), kernel)

    train_ds = TensorDataset(batch_x, batch_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    from_disk_loss, from_disk_mse = test(model, train_dl, loss_fn)

    from_empirical_lip_const = evaluate_lip_const(model=model, x=x, seed=42)
    return (
        mse,
        empirical_lip_const.numpy(),
        from_disk_mse,
        from_empirical_lip_const.numpy(),
    )


class LipschitzLayersTest(unittest.TestCase):
    def _check_mse_results(self, mse, from_disk_mse, test_params):
        self.assertAlmostEqual(
            mse,
            from_disk_mse,
            5,
            "serialization must not change the " "performance of a layer",
        )

    def _check_emp_lip_const(self, emp_lip_const, from_disk_emp_lip_const, test_params):
        self.assertAlmostEqual(
            emp_lip_const,
            from_disk_emp_lip_const,
            5,
            "serialization must not change the " "Lipschitz constant of a layer",
        )
        if test_params["layer_type"] != Linear:
            self.assertLess(
                emp_lip_const,
                test_params["k_lip_model"] * 1.02,
                msg=" the lip const of the network must be lower than the specified boundary",  # noqa: E501
            )

    def _apply_tests_bank(self, tests_bank):
        for test_params in tests_bank:
            pp.pprint(test_params)
            (
                mse,
                emp_lip_const,
                from_disk_mse,
                from_disk_emp_lip_const,
            ) = train_k_lip_model(**test_params)
            print("test mse: %f" % mse)
            print(
                "empirical lip const: %f ( expected %s )"
                % (
                    emp_lip_const,
                    min(test_params["k_lip_model"], test_params["k_lip_data"]),
                )
            )
            self._check_mse_results(mse, from_disk_mse, test_params)
            self._check_emp_lip_const(
                emp_lip_const, from_disk_emp_lip_const, test_params
            )

    def test_vanilla_linear(self):
        """
        Tests for a standard Linear layer, for result comparison.
        """
        self._apply_tests_bank(
            [
                dict(
                    layer_type=Linear,
                    layer_params={
                        "in_features": 4,
                        "out_features": 4,
                        "bias": True,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=None,
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=Linear,
                    layer_params={
                        "in_features": 4,
                        "out_features": 4,
                        "bias": True,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=None,
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=Linear,
                    layer_params={
                        "in_features": 4,
                        "out_features": 4,
                        "bias": True,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=None,
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                ),
            ]
        )

    def test_spectral_linear(self):
        self._apply_tests_bank(
            [
                dict(
                    layer_type=SpectralLinear,
                    layer_params={
                        "in_features": 4,
                        "out_features": 4,
                        "bias": True,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=None,
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=SpectralLinear,
                    layer_params={
                        "in_features": 4,
                        "out_features": 4,
                        "bias": True,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=None,
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=SpectralLinear,
                    layer_params={
                        "in_features": 4,
                        "out_features": 4,
                        "bias": True,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=None,
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                ),
            ]
        )

    def test_frobenius_linear(self):
        self._apply_tests_bank(
            [
                dict(
                    layer_type=FrobeniusLinear,
                    layer_params={
                        "in_features": 4,
                        "out_features": 4,
                        "bias": True,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=None,
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=FrobeniusLinear,
                    layer_params={
                        "in_features": 4,
                        "out_features": 4,
                        "bias": True,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=None,
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=FrobeniusLinear,
                    layer_params={
                        "in_features": 4,
                        "out_features": 4,
                        "bias": True,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=None,
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                ),
            ]
        )

    def test_spectralConv1d(self):
        self._apply_tests_bank(
            [
                dict(
                    layer_type=SpectralConv1d,
                    layer_params={
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": (3, 3),
                        "bias": False,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=SpectralConv1d,
                    layer_params={
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": (3, 3),
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=SpectralConv1d,
                    layer_params={
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": (3, 3),
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                ),
            ]
        )

    def test_spectralConv2d(self):
        self._apply_tests_bank(
            [
                dict(
                    layer_type=SpectralConv2d,
                    layer_params={
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": (3, 3),
                        "bias": False,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=SpectralConv2d,
                    layer_params={
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": (3, 3),
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=SpectralConv2d,
                    layer_params={
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": (3, 3),
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                ),
            ]
        )

    def _test_spectralConv3d(self):  # TBC
        self._apply_tests_bank(
            [
                dict(
                    layer_type=SpectralConv3d,
                    layer_params={
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": (3, 3, 3),
                        "bias": False,
                        "stride": (2, 1, 1),
                        "dilation": (1, 1, 1),
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=SpectralConv3d,
                    layer_params={
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": (3, 3, 3),
                        "stride": (2, 1, 1),
                        "dilation": (1, 1, 1),
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=SpectralConv3d,
                    layer_params={
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": (3, 3, 3),
                        "stride": (2, 1, 1),
                        "dilation": (1, 1, 1),
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                ),
            ]
        )

    def test_frobeniusConv2d(self):
        # tests only checks that lip cons is enforced
        self._apply_tests_bank(
            [
                dict(
                    layer_type=FrobeniusConv2d,
                    layer_params={
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": (3, 3),
                        "bias": False,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=FrobeniusConv2d,
                    layer_params={
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": (3, 3),
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=FrobeniusConv2d,
                    layer_params={
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": (3, 3),
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                ),
            ]
        )

    def _test_ScaledAvgPool2d(self):  # TBC
        self._apply_tests_bank(
            [
                # tests only checks that lip cons is enforced
                dict(
                    layer_type=ScaledAvgPool2d,
                    layer_params={
                        "kernel_size": (2, 2),
                    },
                    batch_size=1000,
                    steps_per_epoch=1,
                    epochs=1,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                ),
                dict(
                    layer_type=ScaledAvgPool2d,
                    layer_params={
                        "kernel_size": (2, 2),
                    },
                    batch_size=1000,
                    steps_per_epoch=1,
                    epochs=1,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                ),
                dict(
                    layer_type=ScaledAvgPool2d,
                    layer_params={
                        "kernel_size": (2, 2),
                    },
                    batch_size=1000,
                    steps_per_epoch=1,
                    epochs=1,
                    in_features=4,
                    out_features=4,
                    input_shape=(1, 5, 5),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                ),
            ]
        )


if __name__ == "__main__":
    unittest.main()
