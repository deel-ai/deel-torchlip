:orphan:

.. role:: hidden
    :class: hidden-section

.. _deel-torchlip-api:

deel.torchlip
=============

.. currentmodule:: deel.torchlip

.. toctree::
   :maxdepth: 4

   deel.torchlip.utils
   deel.torchlip.functional
   deel.torchlip.init
   deel.torchlip.normalizers

Containers
----------

.. autoclass:: LipschitzModule
   :members:
.. autoclass:: Sequential


Convolution Layers
------------------

.. autoclass:: SpectralConv2d
.. autoclass:: FrobeniusConv2d
.. autoclass:: SpectralConvTranspose2d

Pooling Layers
--------------

.. autoclass:: ScaledAdaptiveAvgPool2d
.. autoclass:: ScaledAvgPool2d
.. autoclass:: ScaledL2NormPool2d
.. autoclass:: ScaledAdaptativeL2NormPool2d
.. autoclass:: InvertibleDownSampling
.. autoclass:: InvertibleUpSampling

Non-linear Activations
----------------------

.. autoclass:: MaxMin
.. autoclass:: GroupSort
.. autoclass:: GroupSort2
.. autoclass:: FullSort
.. autoclass:: LPReLU

Linear Layers
-------------

.. autoclass:: SpectralLinear
.. autoclass:: FrobeniusLinear

Loss Functions
--------------

.. autoclass:: KRLoss
.. autoclass:: NegKRLoss
.. autoclass:: HingeMarginLoss
.. autoclass:: HKRLoss
.. autoclass:: HKRMulticlassLoss
.. autoclass:: SoftHKRMulticlassLoss
.. autoclass:: TauCrossEntropyLoss
.. autoclass:: TauBCEWithLogitsLoss
.. autoclass:: CategoricalHingeLoss
