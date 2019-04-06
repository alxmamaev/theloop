import torch
from torch import nn
from random import randint
from torch import optim
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm, tqdm_notebook

class TheLoop:
    def __init__(self, model, criterion, batch_callback,
                 val_callback=None,
                 optimizer="Adam",
                 optimizer_params={"lr":1e-4},
                 scheduler=None,
                 scheduler_params={},
                 device="cpu",
                 val_rate=-1,
                 logdir="./logs",
                 name="experiment",
                 loss_key="loss",
                 val_criterion_key=None,
                 val_criterion_mode="max",
                 use_best_model=True,
                 using_tqdm_notebook=False,):


                assert val_criterion_mode in ["max", "min"]

                self.device = torch.device(device)
                self.model = model.to(self.device)

                if type(criterion) == str:
                    criterion = nn.__dict__[criterion]
                if type(optimizer) == str:
                    optimizer = optim.__dict__[optimizer]

                if scheduler is not None and type(scheduler) == str:
                    scheduler = optim.lr_scheduler.__dict__[scheduler]

                self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
                self.criterion = criterion()
                self.batch_callback = batch_callback
                self.val_callback = val_callback
                self.logdir = logdir
                self.val_rate = val_rate
                self.name = name
                self.loss_key = loss_key
                self.val_criterion_key = val_criterion_key
                self.val_criterion_mode = val_criterion_mode
                self.using_tqdm_notebook = using_tqdm_notebook
                self.use_best_model = use_best_model
                if scheduler is not None:
                    self.scheduler = scheduler(self.optimizer, **scheduler_params)
                else:
                    self.scheduler = None

                self.checkpoint_dir = os.path.join(logdir, "checkpoints")
                self.tensorboard_dir = os.path.join(logdir, "tb_log")

    @staticmethod
    def tb_log(writer, data, it):
        for k, v in data.items():
            writer.add_scalar(k, float(v), it)


    def a(self, train_dataset, val_dataset=None, batch_size=32, n_workers=1,
          shuffle=True, n_epoch=10):
        train_dl = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=n_workers, shuffle=shuffle)
        val_dl = None

        if val_dataset is not None:
            val_dl = DataLoader(val_dataset, batch_size=batch_size,
                                num_workers=n_workers, shuffle=shuffle)


        return self.ka(train_dl, val_dl, n_epoch)

    def ka(self, train_dataloader, val_dataloader=None, n_epoch=10):
        it = 0
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        best_checkpoint = None
        best_checkpoint_score = None
        best_checkpoint_validation = None

        train_size = len(train_dataloader)
        exp_id = randint(0, 10000)
        exp_tb_dir = os.path.join(self.tensorboard_dir, str(exp_id))

        writer = SummaryWriter(log_dir=exp_tb_dir, filename_suffix=self.name)

        print("=====================\n||STARTING THE LOOP||\n=====================\n\n")


        for epoch in range(n_epoch):
            if self.scheduler is not None:
                self.scheduler.step()

            print("  |￣￣￣￣￣￣|\n  |  EPOCH: %s  |\n  |＿＿＿＿＿＿|\n(\\__/) || \n(•ㅅ•) || \n/ 　 づ" % epoch)
            if self.using_tqdm_notebook:
                tqdm_dl = tqdm_notebook(train_dataloader)
            else:
                tqdm_dl = tqdm(train_dataloader)

            for i, batch in enumerate(tqdm_dl):
                self.model.train()
                batch_out = self.batch_callback(model=self.model,
                                                criterion=self.criterion,
                                                device=self.device,
                                                batch=batch)

                loss = batch_out[self.loss_key]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.tb_log(writer, batch_out, it)

                tqdm_dl.set_description('BATCH %i; ITER %s' % (i, it))
                tqdm_dl.set_postfix(loss=loss.item())

                if val_dataloader is not None:
                    if (self.val_rate > 0 and it % self.val_rate == 0) or (epoch+1 == n_epoch and i+1 == train_size):
                        print("Starting validation...")

                        self.model.eval()
                        val_out = self.val_callback(model=self.model,
                                                    data=val_dataloader,
                                                    device=self.device)
                        self.model.train()
                        self.tb_log(writer, val_out, it)

                        print("Validation ready!")


                        if self.val_criterion_key is not None:
                            val_score = float(val_out[self.val_criterion_key])

                            if best_checkpoint is None:
                                best_checkpoint = self.model.state_dict()
                                best_checkpoint_score = val_score
                                best_checkpoint_validation = val_out

                            else:
                                if self.val_criterion_mode == "max" and val_score > best_checkpoint_score:
                                    best_checkpoint = self.model.state_dict()
                                    best_checkpoint_score = val_score
                                    best_checkpoint_validation = val_out

                                elif self.val_criterion_mode == "min" and val_score < best_checkpoint_score:
                                    best_checkpoint = self.model.state_dict()
                                    best_checkpoint_score = val_score
                                    best_checkpoint_validation = val_out


                it += 1

            print("Save epoch checkpoint")
            torch.save(self.model.state_dict(),
                        os.path.join(self.checkpoint_dir,
                                     "%s_epoch_%s.pth" %
                                     (self.name, epoch)))
            print("===================================\n\n")


        if val_dataloader is None:
            print("\n\nFINAL METRICS\n==================")
            for k, v in val_out.items():
                print("|| %s: %s" % (k, float(v)))
            print("==================")
        else:
            torch.save(best_checkpoint,
                       os.path.join(self.checkpoint_dir,
                                     "%s_best.pth" %
                                     (self.name,)))
            print("\n\nBEST METRICS\n==================")
            print("|| Best checkpoint score:", best_checkpoint_score)

            for k, v in best_checkpoint_validation.items():
                print("|| %s: %s" % (k, float(v)))
            print("==================")


            if self.use_best_model:
                self.model.load_state_dict(best_checkpoint)

        return self.model
