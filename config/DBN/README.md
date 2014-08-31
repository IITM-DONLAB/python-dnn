Model Config
------------
* `nnetType` : (Mandatory) specify Type of Network (CNN/RBM/SDA/DNN)
* `train_data` : (Mandatory) specify the working directory containing data configuration and output
* `wdir` : (Mandatory) Working Directory.
* `data_spec` : (Mandatory) specify the path of the data sepification relative to `model_config.json`
* `nnet_spec` : (Mandatory) specify the path of network configuration specification relative to `model_config.json`

* `output_file` : (Mandatory) specify the path of RBM network output file relative to working directory
* `input_file` : specify the path of RBM network inpu file relative to working directory

* `batch_size` : specify the mini batch size while training, default 128

* `n_ins` :784
* `n_outs` :10

* `gbrbm_learning_rate` : pretraining
* `pretraining_learning_rate` : pretraining
* `pretraining_epochs` :

* `initial_pretrain_momentum` :Specify the momentum factor while training default 0.5
* `final_pretrain_momentum` :Specify the momentum factor while training default 0.9
* `initial_pretrain_momentum_epoch` : Specify the momentum factor while training default 5

* `finetune_method` :  Two methods are supported  C: Constant learning rate and E:  Exponential decay
* `finetune_rate` : learning rate configuration

>> * `learning_rate` : For Constant learning rate.default value-0.08
>> * `epoch_num` : For Constant learning rate.default value-10
>> * `start_rate` : For Exponential decay.default value-0.08
>> * `scale_by` : For Exponential decay.default value-0.5
>> * `min_derror_decay_start` : For Exponential decay.default value-0.05
>> * `min_derror_stop` : For Exponential decay.default value-0.05
>> * `min_epoch_decay_start` : For Exponential decay.default value-15
>> * `init_error` : For Exponential decay.default value-100

* `finetune_momentum` :  Specify the momentum factor while finetuning

* `processes` :

>> * `pretraining` : default:false
>> * `finetuning` : default:false
>> * `testing` : default:false
>> * `export_data` : default:false

* `export_path` : path (realative to wdir) for writting (bottleneck) features.

