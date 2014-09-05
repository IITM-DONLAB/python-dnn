Model Config
=============
* `nnetType` : (Mandatory) Type of Network (CNN/RBM/SDA/DNN)
* `train_data` : (Mandatory) The working directory containing data configuration and output
* `wdir` : (Mandatory) Working Directory.
* `data_spec` : (Mandatory) The path of the data sepification relative to `model_config.json`
* `nnet_spec` : (Mandatory) The path of network configuration specification relative to `model_config.json`

* `output_file` : (Mandatory) The path of RBM network output file relative to `wdir`
* `input_file` : The path of PreTrained/FineTuned network input file relative to `wdir`.(Mandatory for DNN)

* `random_seed` : Random Seed used for  initialization of weights.
* `logger_level` : Level of Logger.Valid Values are "INFO","DEBUG" and "ERROR"

* `batch_size` : specify the mini batch size while training, default 128
* `n_ins` :Dimension of input (Mandatory for all except CNN)
* `n_outs` :(Mandatory) Dimension of output (No: of Classes) 
* `input_shape`: The input shape of a given feature vector.(Mandatory For CNN).Should be an Array.


* `finetune_params` : Configuration of finetune learning method.Contains a json object with following params:

> * `momentum` :  The momentum factor while finetuning
> * `method` :  Two methods are supported

>> 1. C: **Constant learning rate**(DEFAULT): run `epoch_num` iterations with `learning_rate` unchanged
>> 2. E: **Exponential decay**: we start with the learning rate of `start_rate`; if the validation error reduction between two epochs is less than `min_derror_decay_start`, the learning rate is scaled by `scale_by` during each of the remaining epoch. The whole traing terminates when the validation error reduction between two epochs falls below `min_derror_stop`. `min_epoch_decay_star` is the minimum epoch number after which scaling can only be performed.


> Default value of other paramters

> | Param                   | Default value  | Learning method 
> |:-----------------------|:--------------:|:---------------:
> |`learning_rate`         |0.08            | C
> |`epoch_num`             |10              | C
> |`start_rate`            |0.08            | E
> |`scale_by`              |0.5             | E
> |`min_derror_decay_start`|0.05            | E
> |`min_derror_stop`       |0.05            | E
> |`min_epoch_decay_start` | 15             | E

> These parameters are used by **Constant learning rate** or **Exponential decay**

* `pretrain_params`: Configuration of pretraining method.Contains a json object with following params

> | Param                    | Default value | nnet Type | Description     
> |--------------------------|:-------------:|:---------:|----------------------------------------------------
> | `gbrbm_learning_rate`    |     0.005     |    DBN    | Pretraining learning rate for gbrbm layer.
> | `learning_rate`          |      0.08     |  SDA,DBN  | Pretraining learning rate (DBN: for all layers except gbrbm layer)
> | `epochs`                 |       15      |    DBN    | No of Pretraining epochs
> | `initial_momentum`       |      0.5      |    DBN    | The initial momentum factor while pre-training
> | `final_momentum`         |      0.9      |    DBN    | The final momentum factor while pre-training
> | `initial_momentum_epoch` |       5       |    DBN    | No: of epochs with the initial momentum factor before switching to final momentum factor
> | `keep_layer_num`         |       0       |  SDA,DBN  | From which layer Pre-Trainig Should Start.If non-Zero layer is intilaized with weights from `input_file`

* `export_path` : path (realative to `wdir`) for writting (bottleneck) features.
* `processes` : Process should be run by program.Contains a json object with following params

> * `pretraining` : whether Pre-Training is needed.(invalid for DNN and CNN).(Default value = false)
> * `finetuning` : whether Fine Tuning  is needed.(Default value = false)
> * `testing` : whether Fine Tuning  is needed.(Default value = false)
> * `export_data` : whether extracted features should written to file.If true,`export_path` is required.(Default value = false).


_____________________________________________________________________________________________

**Also See**: 

* [Example-CNN](../sample_config/MNIST/CNN/model_config.json)
* [Example-RBM](../sample_config/MNIST/DBN/model_config.json)
* [Example-SDA](../sample_config/MNIST/SDA/model_config.json)
