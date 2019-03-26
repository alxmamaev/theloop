# theloop 🔄
*This is simpliest and useful train loop for pytorch. You can easy train your model on any dataset and dataloader.*

## Simple intro into-theloop in 30s

```python
from torchvision.datasets import MNIST
import torchvision.models as models
from theloop import TheLoop


train_set = MNIST(root=root, train=True, transform=trans, download=True)
test_set = MNIST(root=root, train=False, transform=trans, download=True)

resnet18 = models.resnet18(pretrained=True)

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
    
    
 
theloop = TheLoop(model, "CrossEntropyLoss", batch_callback,
                  val_callback=val_callback,
                  optimizer_params={"lr": lr},
                  logdir=logdir,
                  checkpoint_dir=checkpoint_dir,
                  val_rate=val_rate,
                  checkpoint_rate=checkpoint_rate,
                  device=device)

 theloop.a(train_set, val_dataset=test_set,
           batch_size=args.batch_size, n_epoch=args.n_epoch)

```
