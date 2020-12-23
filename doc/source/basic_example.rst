Example and usage
=================


In order to make things simple the following rules have been followed during development:

* ``deel-lip`` follows the ``keras`` package structure.
* All elements (layers, activations, initializers, ...) are compatible with standard the ``keras`` elements.
* When a k-Lipschitz layer overrides a standard keras layer, it uses the same interface and the same parameters.
  The only difference is a new parameter to control the Lipschitz constant of a layer.

Which layers are safe to use?
-----------------------------

The following table indicates which layers are safe to use in a Lipshitz network, and which are not.

.. role:: raw-html-m2r(raw)
   :format: html


.. list-table::
   :header-rows: 1

   * - layer
     - 1-lip?
     - torchlip equivalent
     - comments
   * - :class:`Linear`
     - no
     - :class:`.SpectralLinear` \ :raw-html-m2r:`<br>`\ :class:`.FrobeniusLinear`
     - :class:`.SpectralLinear` and :class:`.FrobeniusLinear` are similar when there is a single output.
   * - :class:`Conv2d`
     - no
     - :class:`.SpectralConv2d` \ :raw-html-m2r:`<br>`\ :class:`.FrobeniusConv2d`
     - :class:`.SpectralConv2d` also implements Björck normalization.
   * - :class:`MaxPooling`\ :raw-html-m2r:`<br>`\ :class:`GlobalMaxPooling`
     - yes
     - n/a
     -
   * - :class:`AvgPool2d`\ :raw-html-m2r:`<br>`\ :class:`AdaptiveAvgPool2d`
     - no
     - :class:`.ScaledAvgPool2d`\ :raw-html-m2r:`<br>`\ :class:`.ScaledAdaptiveAvgPool2d`
     - The lipschitz constant is bounded by ``sqrt(pool_h * pool_h)``.
   * - :class:`Flatten`
     - yes
     - n/a
     -
   * - :class:`Dropout`
     - no
     - None
     - The lipschitz constant is bounded by the dropout factor.
   * - :class:`BatchNorm`
     - no
     - None
     - We suspect that layer normalization already limits internal covariate shift.


How to use it?
--------------

Here is a simple example showing how to build a 1-Lipschitz network:

.. code-block:: python

    from deel.torchlip.initializers import BjorckInitializer
    from deel.torchlip.modules.linear import SpectralLinear
    from deel.torchlip.modules.conv SpectralConv2d
    from deel.torchlip.modules.module import Sequential
    from deel.torchlip.modules.activations import PReLUlip
    from torch.nn import MaxPool2d, Flatten, Softmax

    # from tensorflow.keras.layers import Input, Lambda, Flatten, MaxPool2D
    # from tensorflow.keras import backend as K
    # from tensorflow.keras.optimizers import Adam

    # Sequential (resp Model) from deel.model has the same properties as any lipschitz
    # layer ( condense, setting of the lipschitz factor etc...). It act only as a container.
    model = Sequential(
        [
            # Input(shape=(28, 28)),
            # Lipschitz layer preserve the API of their superclass ( here Conv2D )
            # an optional param is available: k_coef_lip which control the lipschitz
            # constant of the layer
            SpectralConv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=(3, 3),
                padding="same",
            ),
            SpectralConv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=(3, 3),
                padding="same",
            ),
            MaxPool2d(kernel_size=(2, 2)),
            SpectralConv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=(3, 3),
                padding="same",
            ),
            SpectralConv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=(3, 3),
                padding="same",
            ),
            MaxPool2d(kernel_size=(2, 2)),
            Flatten(),
            SpectralLinear(256),
            SpectralLinear(10),
            Softmax(),
        ],
        k_coef_lip=0.5,
        name="testing",
    )

    optimizer = Adam(lr=0.001)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

See :ref:`torchlip-api` for a complete API description.
