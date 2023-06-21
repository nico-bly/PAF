import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import numpy as np
from PIL import Image

transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder('D_Trump', transform=transform)

def split_data(dataset):
    test_fake = Subset(dataset, np.arange(100))
    train_fake = Subset(dataset, np.arange(100, 350))
    train_real, test_real = torch.utils.data.random_split(Subset(dataset, np.arange(350, 690)), [240, 100])

    train = torch.cat((torch.stack([transforms.ToTensor()(Image.open(dataset.samples[idx][0])) for idx in train_fake.indices]), 
                       torch.stack([transforms.ToTensor()(Image.open(dataset.samples[idx][0])) for idx in train_real.indices])))

    test = torch.cat((torch.stack([transforms.ToTensor()(Image.open(dataset.samples[idx][0])) for idx in test_fake.indices]), 
                      torch.stack([transforms.ToTensor()(Image.open(dataset.samples[idx][0])) for idx in test_real.indices])))

    train_labels = torch.cat((torch.tensor([0] * len(train_fake)), torch.tensor([1] * len(train_real))))
    test_labels = torch.cat((torch.tensor([0] * len(test_fake)), torch.tensor([1] * len(test_real))))

    idx = torch.randperm(train.shape[0])
    train_shuffle = train[idx]

    return train_shuffle, test

train, test = split_data(dataset)

torchvision.utils.save_image(train[100], "test.png")
torchvision.utils.save_image(test[50], "test2.png")