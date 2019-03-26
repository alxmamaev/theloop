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
    model = kwargs["model"]
    batch = kwargs["batch"]
    device = kwargs["device"]
    criterion = kwargs["criterion"]

    out = model(batch["tokens"].to(device))
    loss = criterion(out, batch["target"].to(device))

    return {"loss": loss}
    
def val_callback(**kwargs):
    model = kwargs["model"]
    dloader = kwargs["data"]
    device = kwargs["device"]

    predict = []
    ground_truth = []

    for batch in tqdm(dloader):
        with torch.no_grad():
            out = F.softmax(model(batch["tokens"].to(device)).cpu(), dim=1)
            pred = torch.argmax(out, dim=1)

        predict += pred.tolist()
        ground_truth += batch["target"].tolist()

    accuracy = accuracy_score(predict, ground_truth)

    print("Accuracy:", accuracy)
    return {"accuracy": accuracy}
    
 theloop = get_trainer(model, args.lr, args.device,
                          args.logdir, args.checkpoint_dir,
                          args.val_rate, args.checkpoint_rate)

    theloop.a(train_dataset, val_dataset=test_dataset,
              batch_size=args.batch_size, n_epoch=args.n_epoch)

```
