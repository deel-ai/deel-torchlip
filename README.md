<!-- Banner section -->
<div align="center">
        <picture>
                <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/banner_dark_torchlip.png">
                <source media="(prefers-color-scheme: light)" srcset="./docs/assets/banner_light_torchlip.png">
                <img alt="DEEL-TORCHLIP Banner" src="./docs/assets/banner_light_torchlip.png">
        </picture>
</div>
<br>

<!-- Badge section -->
<div align="center">
    <a href="https://pypi.org/project/deel-torchlip">
        <img src="https://img.shields.io/pypi/v/deel-torchlip.svg" alt="PyPI">
    </a>
    <a href="https://pypi.org/project/deel-torchlip">
        <img src="https://img.shields.io/pypi/pyversions/deel-torchlip.svg" alt="Python">
    </a>
    <a href="https://deel-ai.github.io/deel-torchlip">
        <img src="https://img.shields.io/badge/doc-url-blue.svg" alt="Documentation">
    </a>
    <a href="https://arxiv.org/abs/2006.06520">
        <img src="https://img.shields.io/badge/arXiv-2006.06520-b31b1b.svg" alt="arXiv">
    </a>
    <a href="https://github.com/deel-ai/deel-torchlip/actions/workflows/python-tests.yml">
        <img src="https://github.com/deel-ai/deel-torchlip/actions/workflows/python-tests.yml/badge.svg?branch=master" alt="Tests">
    </a>
    <a href="https://github.com/deel-ai/deel-torchlip/actions/workflows/python-lints.yml">
        <img src="https://github.com/deel-ai/deel-torchlip/actions/workflows/python-lints.yml/badge.svg?branch=master" alt="Linters">
    </a>
    <a href="https://github.com/deel-ai/deel-torchlip/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/deel-ai/deel-torchlip.svg" alt="License">
    </a>
</div>
<br>


<!-- Short description of your library -->
<p align="center">
  <b>deel-torchlip</b> is an open source Python API to
build and train Lipschitz neural networks. It is built on top of PyTorch.

  <!-- Link to the documentation -->
  <br>
  <a href="https://deel-ai.github.io/deel-torchlip"><strong>Explore deel-torchlip docs »</strong></a>
  <br>

</p>

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

## 📚 Table of contents

- [📚 Table of contents](#-table-of-contents)
- [🔥 Tutorials](#-tutorials)
- [🚀 Quick Start](#-quick-start)
- [📦 What's Included](#-whats-included)
- [👍 Contributing](#-contributing)
- [👀 See Also](#-see-also)
- [🙏 Acknowledgments](#-acknowledgments)
- [👨‍🎓 Creator](#-creator)
- [🗞️ Citation](#-citation)
- [📝 License](#-license)

## 🔥 Tutorials

We propose some tutorials to get familiar with the library and its API:


| Tutorial | Description | Link |
|----------|-------------|------|
| **Wasserstein Tutorial** | Get started with the basics of *torchlip* to compute wasserstein distance. | [![Open In Github](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](docs/notebooks/wasserstein_toy.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/deel-torchlip/blob/master/docs/notebooks/wasserstein_toy.ipynb) |
| **Binary classification** | Learning binary robust classifier with *deel-torchlip*'s API. | [![Open In Github](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](docs/notebooks/wasserstein_classification_MNIST08.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/deel-torchlip/blob/master/docs/notebooks/wasserstein_classification_MNIST08.ipynb) |
| **Tutorial Multiclass classification** | Learning multiclass robust classifier with *deel-torchlip*'s API. | [![Open In Github](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](deel-torchlip/blob/master/docs/notebooks/wasserstein_classification_fashionMNIST.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/deel-torchlip/blob/master/docs/notebooks/wasserstein_classification_fashionMNIST.ipynb) |



## 🚀 Quick Start

The latest release can be installed using `pip`. The `torch` package will also be
installed as a dependency. If `torch` is already present, be sure that the version is
compatible with the deel-torchlip version.

```shell
$ pip install deel-torchlip
```

### Usage

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


## 📦 What's Included

The `deel-torchlip` library proposes a list of 1-Lipschitz layers equivalent to `torch.nn`  ones.

| `torch.nn`  | 1-Lipschitz? | `deel-torchlip` equivalent  | comments   |
|------------------------------------------------------|-------------|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| :class:`torch.nn.Linear`                             | no          | :class:`.SpectralLinear`<br>:class:`.FrobeniusLinear`                                           | :class:`.SpectralLinear` and :class:`.FrobeniusLinear` are similar when there is a single output.                         |
| :class:`torch.nn.Conv2d`                             | no          | :class:`.SpectralConv2d`<br>:class:`.FrobeniusConv2d`                                           | :class:`.SpectralConv2d` also implements Björck normalization.                                                            |
| :class:`torch.nn.Conv1d`                             | no          | :class:`.SpectralConv1d`                                                                        | :class:`.SpectralConv1d` also implements Björck normalization.                                                            |
| :class:`MaxPooling`<br>:class:`GlobalMaxPooling`      | yes         | n/a                                                                                             |                                                                                                                            |
| :class:`torch.nn.AvgPool2d`<br>:class:`torch.nn.AdaptiveAvgPool2d` | no          | :class:`.ScaledAvgPool2d`<br>:class:`.ScaledAdaptiveAvgPool2d`<br>:class:`.ScaledL2NormPool2d`<br>:class:`.ScaledAdaptativeL2NormPool2d` | The Lipschitz constant is bounded by `sqrt(pool_h * pool_w)`.                                                             |
| :class:`Flatten`                                      | yes         | n/a                                                                                             |                                                                                                                            |
| :class:`torch.nn.ConvTranspose2d`                    | no          | :class:`.SpectralConvTranspose2d`                                                               | :class:`.SpectralConvTranspose2d` also implements Björck normalization.                                                   |
| :class:`torch.nn.BatchNorm1d`<br>:class:`torch.nn.BatchNorm2d`<br>:class:`torch.nn.BatchNorm3d` | no          | :class:`.BatchCentering`                                                                        | This layer apply a bias based on statistics on batch, but no normalization factor (1-Lipschitz).                          |
| :class:`torch.nn.LayerNorm`                           | no          | :class:`.LayerCentering`                                                                        | This layer apply a bias based on statistics on each sample, but no normalization factor (1-Lipschitz).                    |
| Residual connections                                  | no          | :class:`.LipResidual`                                                                           | Learn a factor for mixing residual and a 1-Lipschitz branch.                                                              |
| :class:`torch.nn.Dropout`                             | no          | None                                                                                            | The Lipschitz constant is bounded by the dropout factor.                                                                  |

The `deel-torchlip` library proposes a list of classification losses


| Type | `torch.nn`  |  `deel-torchlip` equivalent  | comments   |
|------------------------------------------------------|-------------|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Binary classification          |:class:`torch.nn.BCEWithLogitsLoss`                             |  :class:`.HKRLoss`   | alpha: Regularization factor ([0,1]) between the hinge and the KR loss;   min_margin: Minimal margin for the hinge loss.                    |
| Multiclass classification          |:class:`torch.nn.CrossEntropyLoss`                             |  :class:`.HKRMulticlassLoss`<br>:class:`.SoftHKRMulticlassLoss`                                           |   alpha: Regularization factor ([0,1]) between the hinge and the KR loss;   min_margin: Minimal margin for the hinge loss.     <br> temperature for the softmax calculation    |

## 👍 Contributing


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

## 👀 See Also



More from the DEEL project:

- [Xplique](https://github.com/deel-ai/xplique) a Python library exclusively dedicated to explaining neural networks.
- [deel-lip](https://github.com/deel-ai/deel-lip) a Python library for training k-Lipschitz neural networks on TF.
- [Influenciae](https://github.com/deel-ai/influenciae) Python toolkit dedicated to computing influence values for the discovery of potentially problematic samples in a dataset.
- [oodeel](https://github.com/deel-ai/oodeel) a Python library for post-hoc deep OOD (Out-of-Distribution) detection on already trained neural network image classifiers
- [DEEL White paper](https://arxiv.org/abs/2103.10529) a summary of the DEEL team on the challenges of certifiable AI and the role of data quality, representativity and explainability for this purpose.

## 🙏 Acknowledgments

<div align="right">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://share.deel.ai/apps/theming/image/logo?useSvg=1&v=10"  width="25%" align="right">
    <source media="(prefers-color-scheme: light)" srcset="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png"  width="25%" align="right">
    <img alt="DEEL Logo" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png" width="25%" align="right">
  </picture>
</div>
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.

## 👨‍🎓 Creators


Main contributors of the deel-torchlip library are:

- [Franck Mamalet](mailto:franck.mamalet@irt-saintexupery.com)
- [Corentin Friedrich](mailto:corentin.friedrich@irt-saintexupery.com)
- [Justin Plakoo](mailto:justin.plakoo@irt-saintexupery.com)
- [Thibaut Boissin](mailto:thibaut.boissin@irt-saintexupery.com)
- [Mikael Capelle](mailto:capelle.mikael@gmail.com)
- [Mathieu Serrurier](mailto:mathieu.serrurier@irt-saintexupery.com)

## 🗞️ Citation


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


## 📝 License

The package is released under [MIT license](LICENSE).
