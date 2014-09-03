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

> * `activation` : Activation function used by layers 
> * `use_fast` : if true program will use pylearn2 library for faster computation (Default Value = false)

* `mlp` contains a json object with following parameters:

> * `layers`     : An Array contain size of hidden layers.
> * `activation` : Activation function used by layers

___________________________________________________________________________________
**Also See**

* [Example](../sample_config/MNIST/CNN/nnet_spec.json)
* [Types of Activation functions](Activation_Fns.md)
