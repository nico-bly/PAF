#%%
from sklearn import metrics
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


def collate_fn(batch):
    return {
        'images': torch.stack([x[0] for x in batch]),
        'labels': torch.tensor([x[1] for x in batch])
    }


def accuracy_final(model, test_dl):
    correct = 0
    total = 0
 
    Y_pred =[]
    Y_true =[]
    crap = []
    crap_labels = []

 
    with torch.no_grad():

        for data in test_dl:
            inputs, labels = data["images"], data["labels"]
            

            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
           
            outputs = model(inputs) #feed
            
            predictions = (torch.sigmoid(outputs.reshape(-1)) >= 0.5).float()

            Y_pred.append(predictions[0]) #save prediction
            
            Y_true.append(labels[0]) #save truth
            

            
            total += labels.size(0)
            
            val = (predictions == labels).sum().item()
            # seulement batch =1
            if not val:
                crap.append(inputs[0])
                crap_labels.append(labels[0])
        
            correct += val


    return 100 * correct / total, torch.stack(Y_true),torch.stack(Y_pred),torch.stack(crap), torch.tensor(crap_labels)


def Create_confusion_matrix(Y_true, Y_pred):
    classes = ('Trump_Real', 'Trump_Fake')

    cf_matrix = confusion_matrix(Y_true, Y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=classes, columns=classes)
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_Adam_3000_Trump_Test_Lora.png')

#utilisation de metrics pour etre sur de la matrice de confusion
def Create_confusion_matrix_number_test(Y_true,Y_pred):
    confusion_matrix = metrics.confusion_matrix(Y_true, Y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax)
    plt.savefig('confusion_SD1.5_TestLora_sklearn.png')


def Create_confusion_matrix_number(Y_true, Y_pred, acc):
    classes = ('Trump_Fake', 'Trump_Real')

    cf_matrix = confusion_matrix(Y_true, Y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
    
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 30})
    plt.xlabel('Predicted',fontsize=20)
    plt.ylabel('Actual',fontsize=20)

    # Create the accuracy legend
    accuracy_text = f'Accuracy: {acc:.2f}%'
    plt.text(0.5, 1.08, accuracy_text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,fontsize=20)    
    plt.title('confusion_matrix_Dreambooth_test_Trump',fontsize=20)
    #plt.show()
    plt.savefig('confusion_matrix_Dreambooth_test_Trump.png')



transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_dataset = datasets.ImageFolder(
    os.path.join("data_Trump", "test"), transforms)
class_names = image_dataset.classes
dataloader = torch.utils.data.DataLoader(
    image_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)


model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, 1)

model_ft.load_state_dict(torch.load("models/detection/deepfake_Adam_3600_Human_V4.pth"))
model_ft.eval()
model_ft.to("cuda")

acc, Y_true, Y_pred, crap, crap_labels = accuracy_final(model_ft, dataloader)
Y_true =Y_true.cpu()
Y_pred=Y_pred.cpu()
imshow(torchvision.utils.make_grid(crap.cpu()))
#print([class_names[x] for x in crap_labels.cpu()])
print(acc)
print(metrics.accuracy_score(Y_true, Y_pred))

Create_confusion_matrix_number(Y_true, Y_pred,acc)
#Create_confusion_matrix_number_test( Y_true, Y_pred)