import torch
import torchvision
from torchvision import datasets, models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import numpy as np
from PIL import Image
import os
import torch.optim as optim
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import lr_scheduler


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "data"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

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

def train_model(model, train_dl, num_epochs=25):
    criterion = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    list_loss = []
    list_acc = []

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dl):
            inputs, labels = data
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs.reshape(-1), labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if i % 20 == 19:
                acc = accuracy_score(model_ft, dataloaders["test"])
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f} acc = {acc}')
                list_loss.append(running_loss)
                list_acc.append(acc)
                running_loss = 0.0
        
            #scheduler.step()
    return list_acc, list_loss

def accuracy_score(model, test_dl):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dl:
            inputs,labels = data
            
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            outputs = model(inputs)
            predictions = (outputs.reshape(-1) >= 0.5) * 1
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return 100 * correct / total



model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features

for param in model_ft.parameters():
    param.requires_grad = False

model_ft.fc = nn.Linear(num_ftrs, 1)

model_ft.to("cuda")

acc, loss = train_model(model_ft, dataloaders["train"])

plt.plot(np.arange(len(loss)), loss, 'r', label="loss")
plt.plot(np.arange(len(acc)), acc, 'b', label="accuracy")
plt.legend()
plt.savefig("Acc_Loss_fixed_Adam.png")

torch.save(model_ft.state_dict(), "deepfake_fixed_Adam.pth")

# dataiter = iter(dataloaders["train"])

# imgs, labels = next(dataiter)

# print(labels)

# imgs = imgs.to("cuda")

# outs = model_ft(imgs)

# print(outs)
# model_ft.load_state_dict(torch.load("deepfake.pth"))
# model_ft.eval()

# dataiter = iter(dataloaders["test"])

# imgs, labels = next(dataiter)

# print(labels)

# imgs = imgs.to("cuda")

# outs = model_ft(imgs)

# print(outs)