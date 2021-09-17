.. pytorch lipschitz modules documentation master file, created by
   sphinx-quickstart on Mon Feb 17 16:42:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/deel-ai/torchlip

Welcome to ``torchlip`` documentation!
=======================================

Controlling the Lipschitz constant of a layer or a whole neural network has many applications ranging
from adversarial robustness to Wasserstein distance estimation.

This library provides implementation of **k-Lispchitz layers for** PyTorch.

Content of the library
----------------------

* k-Lipschitz variant of PyTorch layers such as ``Linear``, ``Conv2d`` and ``AvgPool2d``,
* activation functions compatible with ``pytorch``,
* initializers for ``pytorch``,
* loss functions to work with Wasserstein distance estimations.

Installation
------------

You can install ``deel-torchlip`` directly from pypi:

.. code-block:: bash

   pip install deel-torchlip

In order to use ``torchlip``, you also need a `valid pytorch installation <https://pytorch.org/get-started/locally/#installing-on-linux>`_.
``torchlip`` supports torch 1.7.0+.

Cite this work
--------------

.. raw:: html

   This library is the PyTorch equivalent of <a href="https://github.com/deel-ai/deel-lip"><code>deel-lip</code></a>,
   a Tensorflow library built to support the work presented in
   <a href="https://arxiv.org/abs/2006.06520"><i>Achieving robustness in classification using optimal transport with Hinge regularization</i></a>.
   This work can be cited as:

.. code-block:: latex

   @misc{2006.06520,
      Author = {
         Mathieu Serrurier
         and Franck Mamalet
         and Alberto González-Sanz
         and Thibaut Boissin
         and Jean-Michel Loubes
         and Eustasio del Barrio
      },
      Title = {
         Achieving robustness in classification using optimal transport with hinge regularization
      },
      Year = {2020},
      Eprint = {arXiv:2006.06520},
   }

.. toctree::
   :hidden:
   :titlesonly:
   :maxdepth: 4
   :caption: Shortcuts
   :glob:

   self
   basic_example.rst
   wasserstein_toy.rst
   wasserstein_toy_classification.rst
   wasserstein_classification_MNIST08.rst
