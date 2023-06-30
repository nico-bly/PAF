import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn.functional as F
from torch.utils.data import Dataset
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


epsilons = [0, .05, .1, .15, .2, .25, .3]

use_cuda=True
# Set random seed for reproducibility


model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)



# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


model.load_state_dict(torch.load("models/detection/deepfake_Adam_3600_Human_V2.pth", map_location=device))


# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
#############################


#class CustomDataset(Dataset):
#    def __init__(self, root_dir, transform=None):
 #       self.root_dir = root_dir
  #      self.transform = transform
   #     self.images = os.listdir(root_dir)

    #def __len__(self):
     #   return len(self.images)
#
 #   def __getitem__(self, idx):
  #      image_name = self.images[idx]
   #     image_path = os.path.join(self.root_dir, image_name)
    #    image = Image.open(image_path)
     #   
      #  if self.transform:
       #     image = self.transform(image)
        
        #return image, torch.tensor(0)  # Return a dummy target tensor

# Define the path to your custom dataset
#dataset_path = 'data_human_v4/test/Human_Fake'

# Define the transformation you want to apply to your images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create an instance of your custom dataset
#custom_dataset = CustomDataset(dataset_path, transform=transform)

test_data = datasets.ImageFolder('data_human_v2/test', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
# Create a data loader for your custom dataset

#batch_size = 1
#shuffle = True
#test_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)


###############################################""
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)



def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(data)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)
#send to the device
        perturbed_data = perturbed_data.to(device).type(data.dtype)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")

# Save the figure as PNG
figure_path = 'accuracy_vs_epsilon.png'
plt.savefig(figure_path, bbox_inches='tight', pad_inches=0)

# Close the plot to free up resources
plt.close()

print(f"Figure saved as: {figure_path}")