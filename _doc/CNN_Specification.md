CNN Specification
=================

It has 2 parts:
> 1. `cnn`
> 2. `mlp`

Each one contain a json object.`cnn` describes convolution layer configuration and `mlp` describes hidden layer configuration.

* `cnn` contains a json object with following parameters:

> * `layers`: An Array of json objects.Each one decribes a convolution layer which contains:
>> * `convmat_dim` : Dimension of Convolution Weight
>> * `num_filters` : No. of Feature maps
>> * `poolsize`    : Dimension for Max-pooling
>> * `flatten`     : whether to flatten output or not(true for last layer else false)
>> * `update`      : true if weight need to updated during training.
>> * `activation`  : Activation function used by this layer, if not present global activation fuction is used.

> * `activation` : Activation function used by layers (global)
> * `use_fast` : if true program will use pylearn2 library for faster computation (Default Value = false)

* `mlp` contains a json object with following parameters:

> * `layers`        : An Array contain size of hidden layers.
> * `adv_activation`: if maxout/pnorm is used.
>> * `method` : 'maxout','pnorm'.
>> In `maxout`, a pooling of neuron o/p is done based on poolsize.
>> But in `pnorm` output is normalized after pooling.
>> * `pool_size`: pool size
>> * `pnorm_order`: order of normalization (in pnorm)

> * `activation`    : Activation function used by layers. (if adv_activation is used, it sholud be either 'linear','relu' or 'cappedrelu')


___________________________________________________________________________________
**Also See**

* [Example]({{site.githubUrl}}/tree/master/sample_config/MNIST/CNN/nnet_spec.json)
* [Types of Activation functions](#activation-functions)
