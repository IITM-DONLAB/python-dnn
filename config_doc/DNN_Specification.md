DNN Specification
=================

* `hidden_layers` : (Mandatory) An Array contain size of hidden RBM layer.
* `activation` : Activation function used by layers.
* `pretrained_layers` : Number of layers to be pre-trained.(Default Value = Size of `hidden_layers`)
* `random_seed` : Random Seed used for init of weights.

* `max_col_norm` :regularization for hidden layer parameter.(Default Value = null)
* `l1_reg` : regularization for hidden layer parameter.(Default Value = null)
* `l2_reg` : regularization for hidden layer parameter.(Default Value = null)

* `do_maxout` : (Default Value = false)
* `pool_size` : (Default Value = 1)
* `do_pnorm`  : (Default Value = false)
* `pnorm_order` : (Default Value = 1)

* `do_dropout` : (Default Value =false)
* `dropout_factor` : (Default Value =[0.0])
* `input_dropout_factor` : (Default Value =0.0)

___________________________________________________________________________________
**Also See**

* [Example](../sample_config/MNIST/DNN/dnn_spec.json)
* [Types of Activation functions](Activation_Fns.md)
