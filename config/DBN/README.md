==Model Config==
nnetType: (Mandatory) specify Type of Network (CNN,RBM)
train_data : (Mandatory) specify the working directory containing data configuration and output
wdir : wdir
data_spec:(Mandatory) specify the path of the validation data relative to the working directory
nnet_spec: (Mandatory) specify the path of RBM/CNN/DNN/SDA network configuration specification relative to working directory
output_file : (Mandatory) specify the path of RBM network output file relative to working directory
input_file : specify the path of RBM network inpu file relative to working directory
batch_size : specify the mini batch size while training, default 128
n_ins:784
n_outs:10

gbrbm_learning_rate: pretraining
pretraining_learning_rate: pretraining
pretraining_epochs:

initial_pretrain_momentum:Specify the momentum factor while training default 0.5
final_pretrain_momentum:Specify the momentum factor while training default 0.9
initial_pretrain_momentum_epoch : Specify the momentum factor while training default 5


finetune_method:  Two methods are supported  C: Constant learning rate and E : Exponential decay"

finetune_rate : learning rate configuration"
>learning_rate: 0.08
>epoch_num: 10

>start_rate: 0.08
>scale_by: 0.5
>min_derror_decay_start: 0.05
>min_derror_stop: 0.05
>min_epoch_decay_start: 15
>init_error:100

finetune_momentum :  Specify the momentum factor while finetuning"
finetune_momentum : 0.5

processes:
>pretraining":true
>finetuning":true
>testing":true
>export_data":false

