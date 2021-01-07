deel.torchlip.init
==================

The :mod:`deel.torchlip.init` contains functions that can be used to
initialize weights of neural networks layers.
Similar to the functions from :mod:`torch.nn.init`, these functions
are in-place functions as indicated by their trailing ``_``.

.. Warning::
   These initializers are provided for completeness but we recommend
   using :func:`torch.nn.init.orthogonal_` to initialize your weights
   when training Lipschitz neural networks.

Initializers
~~~~~~~~~~~~

.. automodule:: deel.torchlip.init
   :members:
   :undoc-members:
   :show-inheritance:
