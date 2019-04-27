# theloop 🔄
*This is simpliest and useful train loop for pytorch. You can easy train your model on any dataset and dataloader.*

## Installing
```
git submodule add https://github.com/alxmamaev/theloop
git submodule init
```

## Examples 🔬
* [CIFAR10 example](https://github.com/alxmamaev/theloop/blob/master/examples/cifar10.ipynb)
* IMDB example *(in progress)*
* Metric learning example *(in progress)*
* Autoencoder example *(in progress)*

## We have a cool logs 😎
```
   _____ _______       _____ _______   _______ _    _ ______ _      ____   ____  _____
  / ____|__   __|/\   |  __ \__   __| |__   __| |  | |  ____| |    / __ \ / __ \|  __ \
 | (___    | |  /  \  | |__) | | |       | |  | |__| | |__  | |   | |  | | |  | | |__) |
  \___ \   | | / /\ \ |  _  /  | |       | |  |  __  |  __| | |   | |  | | |  | |  ___/
  ____) |  | |/ ____ \| | \ \  | |       | |  | |  | | |____| |___| |__| | |__| | |
 |_____/   |_/_/    \_\_|  \_\ |_|       |_|  |_|  |_|______|______\____/ \____/|_|

 EXPERIMENT NAME: experiment
 EXPERIMENT ID: 4364
 NUM EPOCH: 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


  |￣￣￣￣￣￣|
    EPOCH: 0  
  |＿＿＿＿＿＿|
(\__/) ||
(•ㅅ•) ||
/ 　 づ
BATCH 1562; ITER 1562: 100%|██████████| 1563/1563 [03:59<00:00,  3.99s/it, loss=1.63]
  0%|          | 0/1563 [00:00<?, ?it/s]
+-------------------+
|   EPOCH METRICS   |
+----------+--------+
|  Metric  | Vlaue  |
+----------+--------+
| accuracy | 0.6173 |
+----------+--------+
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

  |￣￣￣￣￣￣|
    THE END  
  |＿＿＿＿＿＿|
(\__/) ||
(•ㅅ•) ||
/ 　 づ
+-------------------+
|    BEST METRICS   |
+----------+--------+
|  Metric  | Vlaue  |
+----------+--------+
| accuracy | 0.6914 |
+----------+--------+
```

## Features 📊
* Cool logs 😎
* Tensorboard logging
* Checkpoint saving
* Validation
* Selecting the best checkpoint
* Scheduler support
* Tqdm notebook support
* Early stopping by `^C`


## Documentation 🗂
### Parameters:
* `model` - your **nn.Module** model
* `criterion` - **string** with name of loss or **loss class**
* `batch_callback` - function of batch callback (look a examples)
* `val_callback` - function of batch callback (look a examples)
* `optimizer` - **string** with name of optimizer or **optimizer class**
* `optimizer_param` - dict of key-value parameters that will be push into optimizer
* `scheduler` - **string** with name of scheduler or **scheduler class**
* `scheduler_params` - dict of key-value parameters that will be push into scheduler
* `device` - **string** with name of acceleration device (default "cpu")
* `val_rate` - **int** rate of iteration when validation was starting
* `logdir` - **string** path to log directory
* `name` - **string** name of your experiment
* `loss_key` - **string** key of loss in dict returned by batch_callback
* `val_criterion_key` - **string** key of validation in dict returned by val_callback
* `val_criterion_mode` - **string** mode of selecting best checkpoint ("max" or "min")
* `use_best_model` - **bool** use best (by validation metric) model as final model
* `use_tqdm_notebook` - **bool** use tqdm_notebook instead tqdm

### theloop.a
*Train model from dataset*

* `train_dataset` - torch **Dataset** for training
* `val_dataset` - torch **Dataset** for validation
* `batch_size` - **int** batch size
* `n_workers`  - **int** number of workers
* `shuffle` - **bool** shuffle dataset
* `n_epoch` - **int** number of epoch

### theloop.ka
*Train model from dataloader*

* `train_dataloader` - torch **DataLoader** for training
* `val_dataloader` - torch **DataLoader** for validation
* `n_epoch` - **int** number of epoch

## To do
* Early stopping by validation metrics
* Pushing images to Tensorboard
* Data parallel
