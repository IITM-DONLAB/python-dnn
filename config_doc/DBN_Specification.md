DBN(RBM) Specification
=====================

* `hidden_layers` : (Mandatory) An Array contain size of hidden RBM layer.
* `activation` : Activation function used by layers.
* `pretrained_layers` : Number of layers to be pre-trained.(Default Value = Size of `hidden_layers`)
* `first_layer_type` : Type for the first layer.It should be either 'bb' (Bernoulli-Bernoulli) or 'gb' (Gaussian-Bernoulli).(Default Value = gb).


___________________________________________________________________________________
**Also See**

* [Example](../sample_config/MNIST/DBN/rbm_spec.json)
* [Types of Activation functions](Activation_Fns.md)
