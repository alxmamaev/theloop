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

## To do
* Early stopping by validation metrics
* Pushing images to Tensorboard
* Data parallel
