SDA Specification
=================

* `hidden_layers` : (Mandatory) An Array contains size of hidden denoising autoencoder layers.
* `activation` : Activation function used by layers
* `corruption_levels` : (Mandatory) An Array contains corruption level for each layer.Size should be equal to size of `hidden_layers`
* `random_seed` : Random Seed used for init of weights.

___________________________________________________________________________________
**Also See**

* [Example](../sample_config/MNIST/SDA/sda_spec.json)
* [Types of Activation functions](Activation_Fns.md)
