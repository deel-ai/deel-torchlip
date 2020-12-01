# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import math
import os
import pprint
import unittest
from collections import OrderedDict

import numpy as np
import torch.autograd as autograd
from deel.lip.pt.layers import (
    FrobeniusConv2d,
    FrobeniusLinear,
    ScaledAvgPool2d,
    SpectralConv2d,
    SpectralLinear,
    TorchLipschitzLayer,
)
from deel.lip.pt.module import Sequential
from deel.lip.pt.utils import evaluate_lip_const
from pytorch_lightning import metrics
from torch import (
    Tensor,
    abs,
    dot,
    from_numpy,
    load,
    manual_seed,
    no_grad,
    save,
    svd,
    utils,
)
from torch.nn import Conv2d, Linear, MSELoss
from torch.nn import Sequential as tSequential
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# from torch.utils.tensorboard import SummaryWriter

# import torch as t

# seed = t.random.seed


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
    # print("len(input_shape) ===========> {}".format(input_shape))
    # print("len(input_shape) ===========> {}".format(len(input_shape)))
    # print("kernel===========> {}".format(kernel))
    # print("batch_size===========> {}".format(batch_size))

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
        # print("batch_x ===========> {}".format(batch_x.shape))
        # print("batch_y===========> {}".format(batch_y.shape))

        return (from_numpy(batch_x), from_numpy(batch_y))


def build_kernel(in_features: tuple, out_features: tuple, k=1.0):
    """
    build a kernel with defined lipschitz factor

    Args:
        input_shape: input shape of the linear function
        output_shape: output shape of the linear function
        k: lipshitz factor of the function

    Returns:
        the kernel for use in the linear_generator

    """
    input_shape = tuple(in_features)
    output_shape = tuple(out_features)
    # print("input_shape ===========> {}".format(input_shape))
    # print("output_shape===========> {}".format(output_shape))
    # print("k===========> {}".format(k))
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
        a keras Model with a single layer.

    """
    if issubclass(layer_type, Sequential):
        print("layer_params ====> {}".format(layer_params))
        model = layer_type(layer_params)
        model.set_klip_factor(k)

        return model
    # print("input_shape ==========>{}".format(input_shape))
    if issubclass(layer_type, TorchLipschitzLayer):
        layer_params["k_coef_lip"] = k
    # print(
    #     "layer_params['input_shape'] ==========>{}".format(layer_params["input_shape"])
    # )
    layer = layer_type(**layer_params)
    assert isinstance(layer, TorchLipschitzLayer) or isinstance(layer, Linear)
    # assert isinstance(layer, Layer)
    # print(input_shape)
    # print(layer.compute_output_shape((32, ) + input_shape))
    return tSequential(layer)


def train(train_dl, model, loss_fn, optimizer, epoch, batch_size):
    # switch to train mode
    # device = "cpu"
    # params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
    # training_generator = utils.data.DataLoader(train_loader, **params)

    for epoch in range(epoch):
        model.train()

        for xb, yb in train_dl:
            # xb = xb.permute(0, 3, 1, 2)
            # yb = yb.permute(0, 3, 1, 2)
            # print("xb ==========>{}".format(xb.shape))
            # print("yb ==========>{}".format(yb.shape))
            pred = model(xb.float())
            # print("pred ==========>{}".format(pred.shape))
            loss = loss_fn(pred, yb.float())

            # loss = Variable(loss, requires_grad = True)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #     tb.add_histogram("weight_orig", model.weight_orig, epoch)
    #     tb.add_histogram("weight_u", model.weight_u, epoch)
    #     tb.add_histogram("weight_v", model.weight_v, epoch)
    # tb.close()


def test(model, train_dl, loss_fn):
    layer = model[0]
    model.eval()
    test_loss = 0
    runnning_mae = 0
    runnning_mse = 0
    # u, sn, v = svd(layer.weight_orig)
    # cos_err_u = 1.0 - abs(dot(layer.weight_u, u[:, 0]))
    # cos_err_v = 1.0 - abs(dot(layer.weight_v, v[:, 0]))
    # print("initial sn: ==============>", sn)
    # print("u-estimate cosine error: ==============>", cos_err_u)
    # print("v-estimate cosine error: ==============>", cos_err_v)
    with no_grad():
        for x, y in train_dl:
            # y = y.permute(0, 3, 1, 2)
            # data, target = data.to(device), target.to(device)
            output = model(x.float())
            test_loss += loss_fn(output, y.float())  # sum up batch loss
            # valid_accuracy(output, y)
            error = abs(output - y).sum().data
            squared_error = ((output - y) * (output - y)).sum().data
            runnning_mae += error
            runnning_mse += squared_error
            u, s_new, v = svd(layer.weight.data, compute_uv=False)
            print("=========> Actual updated spectral norm: {}".format(s_new[0]))

    test_loss /= len(train_dl.dataset)
    # total_valid_accuracy = valid_accuracy.compute()
    mse = math.sqrt(runnning_mse / len(train_dl.dataset))
    # mae = runnning_mae / len(train_dl.dataset)
    # print("test_loss==============> {}".format(test_loss))
    # print("mse==============> {}".format(mse))
    # print("mae==============> {}".format(mae))
    # print("total_valid_accuracy==============> {}".format(total_valid_accuracy))
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
    # clear session to avoid side effects from previous train
    # K.clear_session()
    np.random.seed(42)
    # create the keras model, defin opt, and compile it
    model = generate_k_lip_model(layer_type, layer_params, k_lip_model)

    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = MSELoss(reduction="sum")
    # model.summary()
    # create the synthetic data generator
    # define logging features
    logdir = os.path.join("logs", "torch_lip_layers", "%s" % layer_type.__name__)
    print("logdir {}".format(logdir))
    # tb = SummaryWriter()
    if issubclass(layer_type, Conv2d) or issubclass(layer_type, Sequential):
        output_shape = compute_output_shape(input_shape, model)
        kernel = build_kernel(input_shape, output_shape, k_lip_data)
    else:
        kernel = build_kernel((in_features,), (out_features,), k_lip_data)
    # train model
    if issubclass(layer_type, Conv2d) or issubclass(layer_type, Sequential):
        (batch_x, batch_y) = linear_generator(batch_size, input_shape, kernel)
    else:
        (batch_x, batch_y) = linear_generator(batch_size, (in_features,), kernel)
    # data = next(iter(batch_x))
    # tb.add_graph(model, data)
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
    if issubclass(layer_type, Conv2d) or issubclass(layer_type, Sequential):
        (x, y) = linear_generator(batch_size, input_shape, kernel)
    else:
        (x, y) = linear_generator(batch_size, (in_features,), kernel)

    np.random.seed(42)
    manual_seed(42)
    if issubclass(layer_type, Conv2d) or issubclass(layer_type, Sequential):
        (batch_x, batch_y) = linear_generator(batch_size, input_shape, kernel)
    else:
        (batch_x, batch_y) = linear_generator(batch_size, (in_features,), kernel)
    train_ds = TensorDataset(batch_x, batch_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    loss, mse = test(model, train_dl, loss_fn)
    # loss, mse = model.__getattribute__(EVALUATE)(
    #     linear_generator(batch_size, input_shape, kernel),
    #     steps=10,
    # )
    empirical_lip_const = evaluate_lip_const(model=model, x=x, seed=42)
    # save the model
    model_checkpoint_path = os.path.join(logdir, "model.h5")
    save(model, model_checkpoint_path)

    del model

    model = load(model_checkpoint_path)
    model.eval()
    np.random.seed(42)
    manual_seed(42)
    if issubclass(layer_type, Conv2d) or issubclass(layer_type, Sequential):
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

    def test_vanilla_dense(self):
        """
        Tests for a standard Dense layer, for result comparison.
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

    def _test_spectral_linear(self):
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

    def _test_spectralConv2d(self):
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

    def _test_frobeniusConv2d(self):
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

    def _test_ScaledAvgPool2d(self):
        print("========================")
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
