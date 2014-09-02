DNN Specification
=================

* `hidden_layers` : (Mandatory) An Array contain size of hidden RBM layer.
* `activation` : Activation function used by layers.
* `pretrained_layers` : Number of layers to be pre-trained.(Default Value = Size of `hidden_layers`)
* `random_seed` : Random Seed used for init of weights.

* `max_col_norm` :regularization for hidden layer parameter.(Default Value = null)
* `l1_reg` : regularization for hidden layer parameter.(Default Value = null)
* `l2_reg` : regularization for hidden layer parameter.(Default Value = null)

* `do_maxout` : whether to use max-out or not. (Default Value = false)
* `do_pnorm`  : whether to use p-norm (Default Value = false)
* `pool_size` : The number of units in each max-pooling(or pnorm) group for maxout/pnorm(Default Value = 1)
* `pnorm_order` : The norm order for pnorm.(Default Value = 1)

* `do_dropout` : whether to use dropout or not. (Default Value =false)
* `dropout_factor` : the dropout factors for DNN layers.(One for each hidden layer)(Default Value =[0.0])
* `input_dropout_factor` : The dropout factor for the input features.(Default Value =0.0)

___________________________________________________________________________________
**Also See**

* [Example](../sample_config/MNIST/DNN/dnn_spec.json)
* [Types of Activation functions](Activation_Fns.md)
