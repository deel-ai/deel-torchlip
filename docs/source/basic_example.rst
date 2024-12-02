Example and usage
=================


In order to make things simple the following rules have been followed during development:

* ``deel-torchlip`` follows the ``torch.nn`` package structure.
* When a k-Lipschitz module overrides a standard ``torch.nn`` module, it uses the same
  interface and the same parameters.
  The only difference is a new parameter to control the Lipschitz constant of a layer.

Which modules are safe to use?
------------------------------

Modules from ``deel-torchlip`` are mostly wrappers around initializers and normalization hooks
that ensure their 1-Lipschitz property.
For instance, the :class:`SpectralLinear` module is simply a :class:`torch.nn.Linear` module
, with automatic orthogonal initialization and hooks:

.. code-block:: python

    # This code is about equivalent to SpectralLinear(16, 32)
    m = torch.nn.Linear(16, 32)
    torch.nn.init.orthogonal_(m.weight)
    m.bias.data.fill_(0.0)

    torch.nn.utils.spectral_norm(m, "weight", 3)
    torchlip.utils.bjorck_norm(m, "weight", 15)


The following table indicates which module are safe to use in a Lipschitz network, and which are not.

.. role:: raw-html-m2r(raw)
   :format: html

.. list-table::
   :header-rows: 1

   * - ``torch.nn``
     - 1-Lipschitz?
     - ``deel-torchlip`` equivalent
     - comments
   * - :class:`torch.nn.Linear`
     - no
     - :class:`.SpectralLinear` \ :raw-html-m2r:`<br>`\ :class:`.FrobeniusLinear`
     - :class:`.SpectralLinear` and :class:`.FrobeniusLinear` are similar when there is a single output.
   * - :class:`torch.nn.Conv2d`
     - no
     - :class:`.SpectralConv2d` \ :raw-html-m2r:`<br>`\ :class:`.FrobeniusConv2d`
     - :class:`.SpectralConv2d` also implements Björck normalization.
   * - :class:`MaxPooling`\ :raw-html-m2r:`<br>`\ :class:`GlobalMaxPooling`
     - yes
     - n/a
     -
   * - :class:`torch.nn.AvgPool2d`\ :raw-html-m2r:`<br>`\ :class:`torch.nn.AdaptiveAvgPool2d`
     - no
     - :class:`.ScaledAvgPool2d`\ :raw-html-m2r:`<br>`\ :class:`.ScaledAdaptiveAvgPool2d` \ :raw-html-m2r:`<br>` \ :class:`.ScaledL2NormPool2d` \ :raw-html-m2r:`<br>` \ :class:`.ScaledGlobalL2NormPool2d`
     - The Lipschitz constant is bounded by ``sqrt(pool_h * pool_w)``.
   * - :class:`Flatten`
     - yes
     - n/a
     -   
   * - :class:`torch.nn.ConvTranspose2d`
     - no
     - :class:`.SpectralConvTranspose2d`
     - :class:`.SpectralConvTranspose2d` also implements Björck normalization.
   * - :class:`torch.nn.BatchNorm1d` \ :raw-html-m2r:`<br>` \ :class:`torch.nn.BatchNorm2d` \ :raw-html-m2r:`<br>` \ :class:`torch.nn.BatchNorm3d`
     - no
     - :class:`.BatchCentering`
     - This layer apply a bias based on statistics on batch, but no normalization factor (1-Lipschitz).
   * - :class:`torch.nn.LayerNorm` 
     - no
     - :class:`.LayerCentering`
     - This layer apply a bias based on statistics on each sample, but no normalization factor (1-Lipschitz).
    * - Residual connections 
     - no
     - :class:`.LipResidual`
     - Learn a factor for mixing residual and a 1-Lipschitz branch .
   * - :class:`torch.nn.Dropout`
     - no
     - None
     - The Lipschitz constant is bounded by the dropout factor.

How to use it?
--------------

Here is a simple example showing how to build a 1-Lipschitz network:

.. code-block:: python

    import torch
    from deel import torchlip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # deel-torchlip layers can be used like any torch.nn layers in
    # Sequential or other types of container modules.
    model = torch.nn.Sequential(
        torchlip.SpectralConv2d(1, 32, (3, 3), padding=1),
        torchlip.SpectralConv2d(32, 32, (3, 3), padding=1),
        torch.nn.MaxPool2d(kernel_size=(2, 2)),
        torchlip.SpectralConv2d(32, 32, (3, 3), padding=1),
        torchlip.SpectralConv2d(32, 32, (3, 3), padding=1),
        torch.nn.MaxPool2d(kernel_size=(2, 2)),
        torch.nn.Flatten(),
        torchlip.SpectralLinear(1568, 256),
        torchlip.SpectralLinear(256, 1)
    ).to(device)

    # Training can be done as usual, except that we are doing
    # binary classification with -1 and +1 labels to the target
    # must be fixed from the dataset.
    optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())
    for data, target in mnist_08:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torchlip.functional.hkr_loss(output, target, alpha=10, min_margin=1)
        loss.backward()
        optimizer.step()


See :ref:`deel-torchlip-api` for a complete API description.
