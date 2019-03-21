import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm

class TheLoop:
    def __init__(self, model, optimizer, criterion, batch_callback
                val_callback=None,
                logdir="./logs",
                device="cpu",
                val_rate=-1,
                checkpoint_dir="./checkpoints",
                checkpoint_rate=-1,
                name="experiment"):

                if type(self.criterion) == str:
                    self.criterion = nn.__dict__[criterion]()
                else:
                    self.criterion = criterion

                self.optimizer = optimizer
                self.device = torch.device(device)
                self.model = model.to(self.device)
                self.batch_callback = batch_callback
                self.val_callback = val_callback
                self.logdir = logdir
                self.checkpoint_dir = checkpoint_dir
                self.checkpoint_rate = checkpoint_rate
                self.name = name
                self.loss_key = "loss"
                self.val_rate = val_rate

    @staticmethod
    def tb_log(writer, data, it):
        for k, v in data.items():
            writer.add_scalar(k, v, it)


    def a(self, train_dataset, val_dataset=None, batch_size=32, n_workers=1,
          shuffle=True, n_epoch=10, log_dir=None):
        train_dl = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=n_workers, shuffle=shuffle)
        val_dl = None

        if val_dataset is not None:
            val_dl = DataLoader(val_dataset, batch_size=batch_size,
                                num_workers=n_workers, shuffle=shuffle)


        self.ka(train_dl, val_dl, n_epoch, log_dir)

    def ka(self, train_dataloader, val_dataloader=None, n_epoch=10, log_dir=None):
        it = 0
        writer = SummaryWriter(log_dir=log_dir, filename_suffix=self.name)

        print("STARTING THE LOOP")
        os.makedirs(self.checkpoint_dir, exist_ok=True)


        for epoch in range(n_epoch):
            print("  |￣￣￣￣￣￣|\n  |  EPOCH: %s  |\n  |＿＿＿＿＿＿|\n(\\__/) || \n(•ㅅ•) || \n/ 　 づ" % epoch)
            tqdm_dl = tqdm(train_dataloader)

            for i, batch in enumerate(tqdm_dl):
                batch_out = self.batch_callback(model=self.model,
                                                criterion=self.criterion,
                                                device=self.device,
                                                batch=batch)

                loss = batch_out[self.loss_key].item()
                self.optimizer.zero_grad()
                loss.backward()

                self.tb_log(writer, batch_out, it)

                tqdm_dl.set_description('BATCH %i' % i)
                tqdm_dl.set_postfix(loss=loss.item())

                if self.val_rate > 0 and val_dataloader is not None:
                    if it % self.val_rate == 0:
                        print("Starting validation")

                        val_out = self.val_callback(model=self.model,
                                                    val_dataloader=val_dataloader,
                                                    device=self.device)
                        self.tb_log(writer, val_out, it)

                        print("Validation ready")

                if self.checkpoint_rate > 0 and it % self.checkpoint_rate == 0:
                        print("Save checkpoint")

                        torch.save(self.model.state_dict(),
                                    os.path.join(self.checkpoint_dir,
                                                 "%s_iter_%s_epoch_%s.pth" %
                                                 (self.name, epoch, it)))

                it += 1
