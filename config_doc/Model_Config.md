Model Config
=============
* `nnetType` : (Mandatory) Type of Network (CNN/RBM/SDA/DNN)
* `train_data` : (Mandatory) The working directory containing data configuration and output
* `wdir` : (Mandatory) Working Directory.
* `data_spec` : (Mandatory) The path of the data sepification relative to `model_config.json`
* `nnet_spec` : (Mandatory) The path of network configuration specification relative to `model_config.json`

* `output_file` : (Mandatory) The path of RBM network output file relative to `wdir`
* `input_file` : The path of PreTrained/FineTuned network input file relative to `wdir`.(Mandatory for DNN)

* `logger_level` : Level of Logger.Valid Values are "INFO","DEBUG" and "ERROR"

* `batch_size` : specify the mini batch size while training, default 128
* `n_ins` :Dimension of input (Mandatory for all except CNN)
* `n_outs` :(Mandatory) Dimension of output (No: of Classes) 
* `input_shape`: The input shape of a given feature vector.(Mandatory For CNN).Should be an Array.

* `finetune_method` :  Two methods are supported  
>> 1. C: Constant learning rate.
>> 2. E:  Exponential decay.
* `finetune_rate` : Configuration of learning method.Contains a json object with following params

> param                   | description                           | default value  |learning method 
> :-----------------------|:--------------------------------------|:--------------:|:---------------:
> `learning_rate`         |                                       |0.08            | C
> `epoch_num`             |                                       |10              | C
> `start_rate`            |                                       |0.08            | E
> `scale_by`              |                                       |0.5             | E
> `min_derror_decay_start`|                                       |0.05            | E
> `min_derror_stop`       |                                       |0.05            | E
> `min_epoch_decay_start` |                                       | 15             | E

* `finetune_momentum` :  The momentum factor while finetuning
* `export_path` : path (realative to `wdir`) for writting (bottleneck) features.
* `processes` : Process should be run by program.Contains a json object with following params

> * `pretraining` : whether Pre-Training is needed.(invalid for DNN and CNN).(Default value = false)
> * `finetuning` : whether Fine Tuning  is needed.(Default value = false)
> * `testing` : whether Fine Tuning  is needed.(Default value = false)
> * `export_data` : whether extracted features should written to file.If true,`export_path` is required.(Default value = false).

######Specific to DBN(RBM)######
* `gbrbm_learning_rate` : Pretraining learning rate for gbrbm layer.(Default Value = 0.005)
* `pretraining_learning_rate` : Pretraining learning rate for all layers except gbrbm layer.(Default Value = 0.08)
* `pretraining_epochs` :No of Pretraining epochs(Default Value = 10)
* `initial_pretrain_momentum` :The initial momentum factor while pre-training (Default Value = 0.5)
* `final_pretrain_momentum` :The final momentum factor while pre-training (Default Value = 0.9)
* `initial_pretrain_momentum_epoch` : No: of epochs with the initial momentum factor before switching to final momentum factor.(Default Value = 5)

* `keep_layer_num`: From which layer Pre-Trainig Should Start.(Default Value = 0).If non-Zero layer is intilaized with weights from `input_file`

######Specific to SDA######
* `pretrain_lr` :learning rate to be used during pre-training (Default Value = 0.08).

_____________________________________________________________________________________________________________