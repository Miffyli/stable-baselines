.. _export:


Exporting models
================

After training the agent's model you may want to deploy/use it in some other language
or framework, like PyTorch or `tensorflowjs <https://github.com/tensorflow/tfjs>`_.
Stable Baselines does not include tools to export models to other frameworks, but
this document aims to cover parts that are required for exporting along with
more detailed stories from users of Stable Baselines.


Background
----------

In Stable Baselines the model is stored inside :ref:`policies <policies>` which convert
observations into actions. Each learning algorithm (e.g. DQN, A2C, SAC) contains
one or more policies, some of which are only used for training. Easy way to find
the policy you need is to check the code for the ``predict`` function of the agent:
This function should only call one policy with simple arguments.

Policies hold the necessary Tensorflow placeholders and tensors to do the 
inference (i.e. predict actions), so it is enough to export these policies
to do inference in an another framework. **Note** that learning algorithms also
may contain Tensorflow placeholders, but these are used for training and are
not required for inference.


Export to PyTorch
-----------------

A known working solution is to use :func:`get_parameters <stable_baselines.common.base_class.BaseRLModel.get_parameters>`
function to obtain model parameters, construct network manually in PyTorch and assign parameters correctly. Note that PyTorch
and Tensorflow have internal differences with e.g. 2D convolutions (see discussion linked below).

See `this discussion <https://github.com/hill-a/stable-baselines/issues/372>`_ for details.


Export to tensorflowjs / tfjs
-----------------------------

Can be done via Tensorflow's `simple_save <https://www.tensorflow.org/api_docs/python/tf/saved_model/simple_save>`_ function
and `tensorflowjs_converter <https://www.tensorflow.org/js/tutorials/conversion/import_saved_model>`_. 

See `this discussion <https://github.com/hill-a/stable-baselines/issues/474>`_ for details.


Export to Java
---------------

Can be done via Tensorflow's `simple_save <https://www.tensorflow.org/api_docs/python/tf/saved_model/simple_save>`_ function.

See `this discussion <https://github.com/hill-a/stable-baselines/issues/329>`_ for details.


Manual export
-------------

You can also manually export required parameters (weights) and construct the
network in your desired framework, as done with the PyTorch example above.

You can access model's parameters via agents' 
:func:`get_parameters <stable_baselines.common.base_class.BaseRLModel.get_parameters>`
function. If you use default policies, you can find the architecture of the networks in
source for :ref:`policies <policies>`.
