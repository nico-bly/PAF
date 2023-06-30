import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torchvision
from torchvision import datasets, models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import numpy as np
from PIL import Image
import torch.optim as optim
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import lr_scheduler
import wandb
import itertools



data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
        transforms.RandomRotation(20),
        transforms.GaussianBlur(kernel_size=(9,9), sigma=(0.1,0.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def collate_fn(batch):
    return {
      'images': torch.stack([x[0] for x in batch]),
      'labels': torch.tensor([x[1] for x in batch])
    }

def get_dataloaders(image_datasets, batch_size=32, percentage=1):

    return {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, num_workers=4, collate_fn=collate_fn, shuffle=True) for x in ['test', 'val', 'train']}

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(model, train_dl, val_dl, num_epochs=25):
    criterion = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    i_train = 1
    i_val = 1

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    for epoch in range(num_epochs):
        for data in train_dl:
            model.train()
            inputs_train, labels_train = data["images"], data["labels"]
            inputs_train = inputs_train.to("cuda")
            labels_train = labels_train.to("cuda")

            optimizer.zero_grad()

            outputs_train = model(inputs_train)

            loss_train = criterion(outputs_train.reshape(-1), labels_train.float())
            loss_train.backward()
            optimizer.step()
            
            acc_train = accuracy_score(model, train_dl)

            wandb.log({"acc_train": acc_train, "loss_train": loss_train.item(), "i_train": i_train})
            i_train += 1

        with torch.no_grad():
            model.eval()
            for data in val_dl:
                inputs_val, labels_val = data["images"], data["labels"]
                inputs_val = inputs_val.to("cuda")
                labels_val = labels_val.to("cuda")

                outputs_val = model(inputs_val)

                loss_val = criterion(outputs_val.reshape(-1), labels_val.float())
                acc_val = accuracy_score(model, val_dl)

                wandb.log({"acc_val": acc_val, "loss_val": loss_val.item(), "i_val": i_val})
                i_val += 1

                #print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss_train.item():.3f} acc = {acc}')
            

def accuracy_score(model, dl):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dl:
            inputs, labels = data["images"], data["labels"]
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            outputs = model(inputs)
            #predictions = (torch.sigmoid(outputs.reshape(-1)) >= 0.5).float()
            predictions = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return 100 * correct / total



model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features

# for param in model_ft.parameters():
#     param.requires_grad = False

model_ft.fc = nn.Linear(num_ftrs, 3)

model_ft.to("cuda")

image_datasets = {x: datasets.ImageFolder(os.path.join("data_human_v5", x), data_transforms[x]) for x in ['train', 'test', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'val']}

### RUN 1 : 50% OF DATASET ###

dataloaders = get_dataloaders(image_datasets, batch_size=64, percentage=0.5)

wandb.init(
    project="deepfake-paf",

    config={
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 64
    }
)

train_model(model_ft, dataloaders["train"], dataloaders["val"], num_epochs=5)

wandb.alert(
    title="Training Done For Adam_3600_Human_V5",
    text="DOne"
)

wandb.finish()

torch.save(model_ft.state_dict(), "models/detection/deepfake_Adam_3600_Human_V5_3outputs.pth")

### RUN 2 : 100% OF DATASET ###

# dataloaders = get_dataloaders(image_datasets, batch_size=64)

# wandb.init(
#     project="deepfake-paf",

#     config={
#         "learning_rate": 0.001,
#         "epochs": 5,
#         "batch_size": 64
#     }
# )

# train_model(model_ft, dataloaders["train"], dataloaders["val"], num_epochs=5)

# wandb.alert(
#     title="Training Done For Adam_3000_100%",
#     text="Everything Done !"
# )

# wandb.finish()

# torch.save(model_ft.state_dict(), "deepfake_Adam_3000_100%.pth")