import warnings
import os
import copy
import math
from functools import partial
import torch
from torch.nn import Sequential as tSequential

from torch.optim import SGD as tSGD
from torch.optim import Adam as tAdam
from torch.nn import CrossEntropyLoss
import torch.autograd as autograd
from torch import Tensor
from torch import from_numpy
from torch import manual_seed as set_seed
from torch import no_grad
from torch.nn import MSELoss as MeanSquaredError
from torch.nn import Linear as tLinear
from torch.nn import Flatten
from torch.nn import ReLU as tReLU
from torch.nn import Softmax as tSoftmax
from torch import reshape
from torch.nn import MaxPool2d as tMaxPool2d
from torch.nn import Conv2d as tConv2d
from torch.nn import Conv2d as PadConv2d
from torch.nn import Upsample as tUpSampling2d
from torch import cat as tConcatenate
from torch import int32 as type_int32
from torch.nn.functional import pad

from deel.torchlip import GroupSort
from deel.torchlip import GroupSort2
from deel.torchlip import Sequential
from deel.torchlip.modules import LipschitzModule as LipschitzLayer
from deel.torchlip.modules import SpectralLinear
from deel.torchlip.modules import SpectralConv2d
from deel.torchlip.modules import FrobeniusLinear
from deel.torchlip.modules import FrobeniusConv2d
from deel.torchlip.modules import ScaledAvgPool2d
from deel.torchlip.modules import ScaledAdaptiveAvgPool2d
from deel.torchlip.modules import ScaledL2NormPool2d
from deel.torchlip.modules import InvertibleDownSampling
from deel.torchlip.modules import InvertibleUpSampling
from deel.torchlip.utils import evaluate_lip_const

from deel.torchlip.modules import (
    KRLoss,
    HingeMarginLoss,
    HKRLoss,
    KRMulticlassLoss,
    HingeMulticlassLoss,
    HKRMulticlassLoss,
    SoftHKRMulticlassLoss,
)

from deel.torchlip.init import spectral_, bjorck_
from deel.torchlip.normalizers import spectral_normalization
from deel.torchlip.normalizers import bjorck_normalization
from deel.torchlip.normalizers import DEFAULT_NITER_SPECTRAL_INIT
from deel.torchlip.modules import vanilla_model
from deel.torchlip.functional import invertible_downsample
from deel.torchlip.functional import invertible_upsample

from deel.torchlip.utils.bjorck_norm import bjorck_norm, remove_bjorck_norm
from deel.torchlip.utils.frobenius_norm import (
    frobenius_norm,
    remove_frobenius_norm,
)
from deel.torchlip.utils.lconv_norm import (
    compute_lconv_coef,
    lconv_norm,
    remove_lconv_norm,
)
from torch.nn import Module as Loss


# to avoid linter F401
__all__ = [
    "tLinear",
    "tSequential",
    "set_seed",
    "MeanSquaredError",
    "Flatten",
    "tSoftmax",
    "tMaxPool2d",
    "tConv2d",
    "tUpSampling2d",
    "tConcatenate",
    "type_int32",
    "GroupSort",
    "GroupSort2",
    "Sequential",
    "FrobeniusLinear",
    "FrobeniusConv2d",
    "InvertibleDownSampling",
    "InvertibleUpSampling",
    "evaluate_lip_const",
    "DEFAULT_NITER_SPECTRAL_INIT",
    "invertible_downsample",
    "invertible_upsample",
    "bjorck_norm",
    "remove_bjorck_norm",
    "frobenius_norm",
    "remove_frobenius_norm",
    "compute_lconv_coef",
    "lconv_norm",
    "remove_lconv_norm",
    "Loss",
]


# not implemented
def module_Unavailable(**kwargs):
    return None


def module_Unavailable_foo(foo=None, **kwargs):
    return None


class module_Unavailable_class:
    def __init__(self, foo=None, **kwargs):
        self.unavailable = True
        return None

    def unavailable_class():
        return True

    def __call__(self, **kwargs):
        return None


tInput = module_Unavailable_foo
Householder = module_Unavailable_class
SpectralConv2dTranspose = module_Unavailable_class
ScaledGlobalL2NormPool2d = module_Unavailable_class
AutoWeightClipConstraint = module_Unavailable_class
SpectralConstraint = module_Unavailable_class
FrobeniusConstraint = module_Unavailable_class
CondenseCallback = module_Unavailable_class
MonitorCallback = module_Unavailable_class
MultiMarginLoss = module_Unavailable_class
TauCategoricalCrossentropyLoss = module_Unavailable_class
TauSparseCategoricalCrossentropyLoss = module_Unavailable_class
TauBinaryCrossentropyLoss = module_Unavailable_class
CategoricalHingeLoss = module_Unavailable_class
process_labels_for_multi_gpu = module_Unavailable_class
CategoricalProvableRobustAccuracy = module_Unavailable_class
BinaryProvableRobustAccuracy = module_Unavailable_class
CategoricalProvableAvgRobustness = module_Unavailable_class
BinaryProvableAvgRobustness = module_Unavailable_class
reshaped_kernel_orthogonalization = module_Unavailable_class
spectral_normalization_conv = module_Unavailable_class
_padding_circular = module_Unavailable_class
Lorth2d = module_Unavailable_class
LorthRegularizer = module_Unavailable_class
Model = module_Unavailable_class
compute_layer_sv = module_Unavailable_class
OrthLinearRegularizer = module_Unavailable_class

MODEL_PATH = "model.h5"
LIP_LAYERS = "torchlip_layers"


def get_instance_generic(instance_type, inst_params):
    return instance_type(**inst_params)


def replace_key_params(inst_params, dict_keys_replace):
    layp = copy.deepcopy(inst_params)
    for k, v in dict_keys_replace.items():
        if k in layp:
            val = layp.pop(k)
            if v is None:
                warnings.warn(
                    UserWarning("Warning key is not used", k, " in tensorflow")
                )
            else:
                if isinstance(v, tuple):
                    layp[v[0]] = v[1](val)
                else:
                    layp[v] = val

    return layp


def get_instance_withreplacement(instance_type, inst_params, dict_keys_replace):
    layp = replace_key_params(inst_params, dict_keys_replace)
    return instance_type(**layp)


def get_instance_withcheck(
    instance_type, inst_params, dict_keys_replace={}, list_keys_notimplemented=[]
):
    for k in list_keys_notimplemented:
        if k in inst_params:
            warnings.warn(
                UserWarning("Warning key is not implemented", k, " in pytorch")
            )
            return None
    layp = replace_key_params(inst_params, dict_keys_replace)
    return instance_type(**layp)


getters_dict = {
    ScaledAdaptiveAvgPool2d: partial(
        get_instance_withreplacement, dict_keys_replace={"data_format": None}
    ),
    ScaledL2NormPool2d: partial(
        get_instance_withreplacement, dict_keys_replace={"data_format": None}
    ),
    SpectralConv2d: partial(
        get_instance_withreplacement, dict_keys_replace={"name": None}
    ),
    SpectralLinear: partial(
        get_instance_withreplacement, dict_keys_replace={"name": None}
    ),
    ScaledAvgPool2d: partial(
        get_instance_withreplacement, dict_keys_replace={"data_format": None}
    ),
    KRLoss: partial(get_instance_withcheck, list_keys_notimplemented=["reduction"]),
    HingeMarginLoss: partial(
        get_instance_withcheck, list_keys_notimplemented=["reduction"]
    ),
    HKRLoss: partial(get_instance_withcheck, list_keys_notimplemented=["reduction"]),
    HingeMulticlassLoss: partial(
        get_instance_withcheck, list_keys_notimplemented=["reduction"]
    ),
    HKRMulticlassLoss: partial(
        get_instance_withcheck, list_keys_notimplemented=["reduction"]
    ),
    KRMulticlassLoss: partial(
        get_instance_withcheck, list_keys_notimplemented=["reduction"]
    ),
    SoftHKRMulticlassLoss: partial(
        get_instance_withcheck, list_keys_notimplemented=["reduction"]
    ),
    tLinear: partial(
        get_instance_withcheck,
        dict_keys_replace={"kernel_initializer": None},
        list_keys_notimplemented=["kernel_regularizer"],
    ),
    spectral_normalization: partial(
        get_instance_withreplacement, dict_keys_replace={"eps": None}
    ),
    bjorck_normalization: partial(
        get_instance_withreplacement, dict_keys_replace={"eps": None}
    ),
    PadConv2d: partial(
        get_instance_withreplacement,
        dict_keys_replace={
            "padding_mode": (
                "padding_mode",
                lambda x: "zeros" if x.lower() in ["same", "valid"] else x,
            ),
        },
    ),
}


def get_instance_framework(instance_type, inst_params):
    if instance_type not in getters_dict:
        instance = get_instance_generic(instance_type, inst_params)
    else:
        instance = getters_dict[instance_type](instance_type, inst_params)
    if instance is None:
        print("instance is not implemented", instance_type)
    return instance


def generate_k_lip_model(layer_type: type, layer_params: dict, input_shape=None, k=1):
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
    if issubclass(layer_type, tSequential):
        layers_list = [lay for lay in layer_params["layers"] if lay is not None]
        assert len(layers_list) > 0
        if k is not None:
            model = layer_type(*layers_list, k_coef_lip=k)
        else:
            model = layer_type(*layers_list)
        # model.set_klip_factor(k)
        return model
    if issubclass(layer_type, LipschitzLayer):
        layer_params["k_coef_lip"] = k

    layer = get_instance_framework(layer_type, layer_params)
    assert isinstance(layer, torch.nn.Module)  # or isinstance(layer, Linear)
    return tSequential(layer)


class tModel(torch.nn.Module):
    def __init__(self, dict_tensors={}, functional_input_output_tensors={}):
        super(tModel, self).__init__()
        self.dict_tensors = dict_tensors
        self.functional_input_output_tensors = functional_input_output_tensors
        self.modList = torch.nn.ModuleList([dict_tensors[key] for key in dict_tensors])

    def forward(self, x):
        x = self.functional_input_output_tensors(self.dict_tensors, x)
        return x


def get_functional_model(modeltype, dict_tensors, functional_input_output_tensors):
    return modeltype(dict_tensors, functional_input_output_tensors)


def compute_output_shape(in_size, mod):
    """
    Compute output size of Module `mod` given an input with size `in_size`.
    """

    f = mod.forward(autograd.Variable(Tensor(1, *in_size)))
    return tuple(f.shape[1:])


# compute_loss
def compute_loss(loss, y_pred, y_true):
    return loss(y_pred, y_true)


def compute_predict(model, x, training=False):
    if not training:
        model.eval()
    return model(x)


def train(
    train_dl,
    model,
    loss_fn,
    optimizer,
    epoch,
    batch_size,
    steps_per_epoch,
    callbacks=[],
):
    for epoch in range(epoch):
        model.train()
        for _ in range(steps_per_epoch):
            # for xb, yb in train_dl:
            xb, yb = next(train_dl)
            xb, yb = from_numpy(xb), from_numpy(yb)
            pred = model(xb.float())
            loss = loss_fn(pred, yb.float())
            # compute gradient and do SGD step
            if optimizer is not None:  # in case of no parameter in the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


def run_test(model, test_dl, loss_fn, metrics, steps=10):
    model.eval()
    test_loss = 0
    for metric in metrics:
        metric.reset_states()
    with no_grad():
        for _ in range(steps):
            # for x, y in test_dl:
            x, y = next(test_dl)
            x, y = from_numpy(x), from_numpy(y)
            output = model(x.float())
            test_loss += loss_fn(output, y.float())  # sum up batch loss
            for metric in metrics:
                metric(y, output)

    test_loss /= steps
    if len(metrics) == 0:
        mse = None
    else:
        mse = metrics[0].result()
    # mse = math.sqrt(runnning_mse / steps)
    return test_loss, mse


def init_session():
    return


def SGD(lr=0.01, momentum=0.0, nesterov=False, model=None):
    if len(list(model.parameters())) == 0:
        return None
    return tSGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)


def Adam(lr=0.01, model=None):
    if len(list(model.parameters())) == 0:
        return None
    return tAdam(model.parameters(), lr=lr)


def CategoricalCrossentropy(from_logits=True):
    assert from_logits, "from_logits has to be True"
    return CrossEntropyLoss()


def build_named_sequential(modeltype, dict_layers):
    return modeltype(dict_layers)


def compile_model(model, loss, optimizer, metrics=[]):
    return loss, optimizer, metrics


def build_layer(layer, input_shape):
    return
    # layer.build(tf.TensorShape((None,) + input_shape))


def to_tensor(nparray, dtype=torch.float32):
    return torch.tensor(nparray, dtype=dtype)


def to_numpy(tens):
    if isinstance(tens, torch.Tensor):
        return tens.detach().numpy()
    else:
        return tens


def save_model(model, path, overwrite=True):
    parent_dirpath = os.path.split(path)[0]
    if not os.path.exists(parent_dirpath):
        os.makedirs(parent_dirpath)
    torch.save(model.state_dict(), path)
    return


def load_model(
    path, compile=False, layer_type=None, layer_params=True, input_shape=None, k=None
):
    model = generate_k_lip_model(layer_type, layer_params, input_shape, k)
    model.load_state_dict(torch.load(path))
    return model


def get_layer_weights_by_index(model, layer_idx):
    return get_layer_weights(model[layer_idx])


# .weight.detach().cpu().numpy()


def get_layer_weights(layer):
    return layer.weight


def get_children(model):
    return model.children()


def get_named_children(model):
    return model.named_children()


def initialize_kernel(model, layer_idx, kernel_initializer):
    kernel_initializer(model[layer_idx].weight)
    return


def initializers_Constant(value):
    return None


def check_parametrization(m, is_parametrized):
    if is_parametrized:
        # ensure that the original weight is the only torch parameter
        assert isinstance(m.parametrizations.weight.original, torch.nn.Parameter)
        assert not isinstance(m.weight, torch.nn.Parameter)
    else:
        assert not hasattr(m, "parametrizations")
        assert isinstance(m.weight, torch.nn.Parameter)


#    return torch.nn.init.constant_(value)


class metric_mse:
    def __init__(self):
        self.mse = 0
        self.count = 0

    def update_state(self, y_true, y_pred):
        self.mse += ((y_true - y_pred) * (y_true - y_pred)).sum().data
        self.count += 1

    def result(self):
        return math.sqrt(self.mse / self.count)

    def reset_states(self):
        self.mse = 0
        self.count = 0

    def __call__(self, y_true, y_pred):
        return self.update_state(y_true, y_pred)


def to_framework_channel(x):
    return x


def to_NCHW(x):
    return x


def get_NCHW(x):
    return (x.shape[0], x.shape[1], x.shape[2], x.shape[3])


def scaleAlpha(alpha):
    # from KR + apha*Hinge to (1-alpha')*KR + alpha'*Hinge
    warnings.warn("scaleAlpha is deprecated, use alpha in [0,1] instead")
    # return 1.0
    return 1.0 / (alpha + 1.0)


def scaleDivAlpha(alpha):
    # soft HKR in TF has a factor 1/alpha*KR + Hinge
    warnings.warn("scaleAlpha is deprecated, use alpha in [0,1] instead")
    # return 1.0
    return 1.0 / (1 + 1.0 / alpha)


def SpectralInitializer(eps_spectral, eps_bjorck):
    warnings.warn("spectral_ and bjorck_ require n_iterations in pytorch")
    if eps_bjorck is None:
        return spectral_
    else:
        return bjorck_


class tAdd(torch.nn.Module):
    def __init__(self):
        super(tAdd, self).__init__()

    def forward(self, x):
        return x[0] + x[1]


def tActivation(activation):
    if activation == "relu":
        return tReLU()
    else:
        return None


class tReshape(torch.nn.Module):
    def __init__(self, target_shape):
        super(tReshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return reshape(x, self.target_shape)


def vanilla_require_a_copy():
    return True


def copy_model_parameters(model_src, model_dest):
    model_dest.load_state_dict(model_src.state_dict())


def vanillaModel(model):
    vanilla_model(model)
    return model


def is_supported_padding(padding):
    return padding.lower() in ["same", "valid", "reflect", "circular"]  # "constant",


def pad_input(x, padding, kernel_size):
    """Pad an input tensor x with corresponding padding, based on kernel size."""
    if isinstance(kernel_size, (int, float)):
        kernel_size = [kernel_size, kernel_size]
    if padding.lower() in ["same", "valid"]:
        return x
    elif padding.lower() in ["constant", "reflect", "circular"]:
        p_vert, p_hor = kernel_size[0] // 2, kernel_size[1] // 2
        pad_sizes = [
            p_hor,
            p_hor,
            p_vert,
            p_vert,
        ]  # [[0, 0], [p_vert, p_vert], [p_hor, p_hor], [0, 0]]
        return pad(x, tuple(pad_sizes), padding)
