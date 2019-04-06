# theloop 🔄
*This is simpliest and useful train loop for pytorch. You can easy train your model on any dataset and dataloader.*

## Simple intro into-theloop in 30s

```python
from theloop import TheLoop

import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrica import accuracy_score

def batch_callback(**kwargs):
    model, batch, device, criterion = kwargs["model"], kwargs["batch"], kwargs["device"], kwargs["criterion"]

    out = model(batch["tokens"].to(device))
    loss = criterion(out, batch["target"].to(device))

    return {"loss": loss}
    
def val_callback(**kwargs):
    model, dloader, device = kwargs["model"], kwargs["data"], kwargs["device"]

    predict = []
    ground_truth = []

    for batch in tqdm(dloader):
        with torch.no_grad():
            out = F.softmax(model(batch["tokens"].to(device)).cpu(), dim=1)
            pred = torch.argmax(out, dim=1)

        predict += pred.tolist()
        ground_truth += batch["target"].tolist()

    accuracy = accuracy_score(predict, ground_truth)

    return {"accuracy": accuracy}
    
    
    
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = MNIST(root=root, train=True, transform=trans, download=True)
test_set = MNIST(root=root, train=False, transform=trans, download=True)

resnet18 = models.resnet18(pretrained=True)

 
theloop = TheLoop(model, "CrossEntropyLoss", batch_callback,
                  val_callback=val_callback,
                  optimizer_params={"lr": 1e-4},
                  logdir="./logdir",
                  checkpoint_dir="./checkpoint_dir",
                  val_rate=1000,
                  checkpoint_rate=1000,
                  device="cuda:0")

theloop.a(train_set, val_dataset=test_set, batch_size=32, n_epoch=10)

```

## Features

### Cool logs
```

=====================
||STARTING THE LOOP||
=====================


  |￣￣￣￣￣￣|
  |  EPOCH: 0  |
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
