![deel-torchlip](docs/source/logo.svg#gh-light-mode-only)
![deel-torchlip](docs/source/logo_white.svg#gh-dark-mode-only)

[![PyPI](https://img.shields.io/pypi/v/deel-torchlip.svg)](https://pypi.org/project/deel-torchlip)
[![Python](https://img.shields.io/pypi/pyversions/deel-torchlip.svg)](https://pypi.org/project/deel-torchlip)
[![License](https://img.shields.io/github/license/deel-ai/deel-torchlip.svg)](https://github.com/deel-ai/deel-torchlip/blob/master/LICENSE)

[![Documentation](https://img.shields.io/badge/doc-url-blue.svg)](https://deel-ai.github.io/deel-torchlip)
[![arXiv](https://img.shields.io/badge/arXiv-2006.06520-b31b1b.svg)](https://arxiv.org/abs/2006.06520)

[![Tests](https://github.com/deel-ai/deel-torchlip/actions/workflows/python-tests.yml/badge.svg?branch=master)](https://github.com/deel-ai/deel-torchlip/actions/workflows/python-tests.yml)
[![Linters](https://github.com/deel-ai/deel-torchlip/actions/workflows/python-lints.yml/badge.svg?branch=master)](https://github.com/deel-ai/deel-torchlip/actions/workflows/python-lints.yml)

[deel-torchlip](https://deel-ai.github.io/deel-torchlip) is an open source Python API to
build and train Lipschitz neural networks. It is built on top of PyTorch.

deel-torchlip provides:

- **Easy-to-use Lipschitz layers** -- deel-torchlip layers are custom PyTorch layers and
  are very user-friendly. No need to be an expert in Lipschitz networks!
- **Custom losses for robustness** -- The provided losses help improving adversarial
  robustness in classification tasks by increasing margins between outputs of the
  network (see [our paper](https://arxiv.org/abs/2006.06520) for more information).
- **Certified robustness** -- One main advantage of Lipschitz networks is the costless
  computation of certificates ensuring that there is no adversarial attacks smaller than
  these certified radii of robustness.

For TensorFlow/Keras users, we released the
[deel-lip](https://deel-lip.readthedocs.io/en/latest/) package offering a similar
implementation based on Keras.

## Install

The latest release can be installed using `pip`. The `torch` package will also be
installed as a dependency. If `torch` is already present, be sure that the version is
compatible with the deel-torchlip version.

```shell
$ pip install deel-torchlip
```

## Usage

Creating a Lipschitz network is similar to building a PyTorch model: standard layers are
replaced with their Lipschitz counterparts from deel-torchlip. PyTorch layers that are
already Lipschitz can still be used in Lipschitz networks, such as `torch.nn.ReLU()` or
`torch.nn.Flatten()`.

```python
import torch
from deel import torchlip

# Build a Lipschitz network with 4 layers, that can be used in a training loop,
# like any torch.nn.Sequential network
model = torchlip.Sequential(
    torchlip.SpectralConv2d(
        in_channels=3, out_channels=16, kernel_size=(3, 3), padding="same"
    ),
    torchlip.GroupSort2(),
    torch.nn.Flatten(),
    torchlip.SpectralLinear(15544, 64)
)
```

See the [full documentation](https://deel-ai.github.io/deel-torchlip) for a complete API
description and for our tutorials to get started.

## Citation

This library was built to support the work presented in our CVPR 2021 paper
[_Achieving robustness in classification using optimal transport with Hinge regularization_](https://arxiv.org/abs/2006.06520).
If you use our library for your work, please cite our paper :wink:

```latex
@misc{2006.06520,
Author = {Mathieu Serrurier and Franck Mamalet and Alberto González-Sanz and Thibaut Boissin and Jean-Michel Loubes and Eustasio del Barrio},
Title = {Achieving robustness in classification using optimal transport with hinge regularization},
Year = {2020},
Eprint = {arXiv:2006.06520},
}
```

## Contributions

Contributions are welcome! You can open an
[issue](https://github.com/deel-ai/deel-torchlip/issues) or fork this repository and
propose a [pull-request](https://github.com/deel-ai/deel-torchlip/pulls). The
development environment with all required dependencies should be installed by running:

```shell
$ make prepare-dev
```

Code formatting and linting are performed with `black` and `flake8`. Tests are run with
`pytest`. These three commands are gathered in:

```shell
$ make test
```

Finally, commits should respect pre-commit hooks. To be sure that your code changes are
accepted, you can run the following target:

```shell
$ make check_all
```

## License

Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry, CRIAQ
and ANITI - https://www.deel.ai/

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgments

This project received funding from the French "Investing for the Future – PIA3" program
within the Artiﬁcial and Natural Intelligence Toulouse Institute (ANITI). The authors
gratefully acknowledge the support of the [DEEL project](https://www.deel.ai/).
