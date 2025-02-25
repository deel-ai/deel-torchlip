.. role:: hidden
    :class: hidden-section

deel.torchlip.functional
========================

.. currentmodule:: deel.torchlip.functional

Non-linear activation functions
-------------------------------

:hidden:`max_min`
~~~~~~~~~~~~~~~~~

.. autofunction:: max_min

:hidden:`group_sort`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: group_sort
.. autofunction:: group_sort_2
.. autofunction:: full_sort

:hidden:`others`
~~~~~~~~~~~~~~~~

.. autofunction:: lipschitz_prelu
    
Padding functions
-------------------------------
.. autoclass:: SymmetricPad

Loss functions
--------------

:hidden:`Binary losses`
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: kr_loss
.. autofunction:: neg_kr_loss
.. autofunction:: hinge_margin_loss
.. autofunction:: hkr_loss

:hidden:`multiclass losses`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hinge_multiclass_loss
.. autofunction:: hkr_multiclass_loss

:hidden:`others`
~~~~~~~~~~~~~~~~

.. autofunction:: process_labels_for_multi_gpu