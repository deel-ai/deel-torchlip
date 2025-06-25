:orphan:

.. role:: hidden
    :class: hidden-section

.. _deel-torchlip-api:

deel.torchlip
=============

.. currentmodule:: deel.torchlip



Containers
----------

.. autoclass:: LipschitzModule
   :members:
.. autoclass:: Sequential


Linear Layers
-------------

.. autoclass:: SpectralLinear
.. autoclass:: FrobeniusLinear

Convolution Layers
------------------

.. autoclass:: SpectralConv1d
.. autoclass:: SpectralConv2d
.. autoclass:: FrobeniusConv2d
.. autoclass:: SpectralConvTranspose2d

Pooling Layers
--------------

.. autoclass:: ScaledAvgPool2d
.. autoclass:: ScaledL2NormPool2d
.. autoclass:: ScaledAdaptiveAvgPool2d
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
.. autoclass:: HouseHolder


Loss Functions
--------------

.. autoclass:: KRLoss
.. autoclass:: NegKRLoss
.. autoclass:: HingeMarginLoss
.. autoclass:: HKRLoss
.. autoclass:: HKRMulticlassLoss
.. autoclass:: SoftHKRMulticlassLoss
.. autoclass:: LseHKRMulticlassLoss
.. autoclass:: TauCrossEntropyLoss
.. autoclass:: TauBCEWithLogitsLoss
.. autoclass:: CategoricalHingeLoss


.. toctree::
   :maxdepth: 4
   
   deel.torchlip.utils
   deel.torchlip.functional
   deel.torchlip.normalizers