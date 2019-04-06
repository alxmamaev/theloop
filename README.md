# theloop 🔄
*This is simpliest and useful train loop for pytorch. You can easy train your model on any dataset and dataloader.*


## Examples 
* [CIFAR10 example](https://github.com/alxmamaev/theloop/blob/master/examples/cifar10.ipynb)
* IMDB example *(in progress)*
* Metric learning example *(in progress)*
* Autoencoder example *(in progress)*

## We have a cool logs 😎
```

=====================
||STARTING THE LOOP||
=====================


  |￣￣￣￣￣￣|
  | EPOCH: 0 |
  |＿＿＿＿＿＿|
(\__/) || 
(•ㅅ•) || 
/ 　 づ
BATCH 1562; ITER 1562: 100%|██████████| 1563/1563 [04:11<00:00,  4.10s/it, loss=1.19]
Validation ready!
Epoch checkpoint saved
===================================

==================
||FINAL METRICS
|| accuracy: 0.6231
==================
```

## Features
* Cool logs 😎
* Tensorboard logging
* Checkpoint saving
* Validation
* Selecting the best checkpoint
* Scheduler support
* Tqdm notebook support
* Early stopping by `^C`


## Doc

## TheLoop
Parameters: 
* *model* - your **nn.Module** model
* *criterion - **string** with name of loss or **loss class**
* batch_callback - function of batch callback (look a examples)
* val_callback - function of batch callback (look a examples)
* optimizer - **string** with name of optimizer or **optimizer class**
* optimizer_param - dict of key-value parameters that will be push into optimizer
* scheduler **string** with name of scheduler or **scheduler class**
* scheduler_params - dict of key-value parameters that will be push into scheduler
* device - **string** with name of acceleration device (default "cpu")
* val_rate - **int** rate of iteration when validation was starting
* logdir - **string** path to log directory
* name - **string** name of your experiment
* loss_key - key of loss in dict returned by batch_callback 
* val_criterion_key - key of validation in dict returned by val_callback 
* val_criterion_mode - mode of selecting best checkpoint ("max" or "min")
* using_tqdm_notebook - use tqdm_notebook instead tqdm 

## To do
* Early stopping by validation metrics
* Pushing images to Tensorboard
* Data parallel
