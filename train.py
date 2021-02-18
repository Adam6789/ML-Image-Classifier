#import model
#import utilities
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json

''' 

1. LOAD THE DATA

'''


parser = argparse.ArgumentParser(
    description='This is a program to train a new network on a dataset of images',
)
# positional arguments
parser.add_argument( action="store", dest='data_dir')
# optional arguments
parser.add_argument('--save_dir', action="store", dest='save_dir', default='model.pth')
parser.add_argument('--arch', action="store", dest='arch', choices={'vgg16','densenet121'},default='densenet121')
parser.add_argument('--learn_rate', action="store", dest='learn_rate', default=0.003)
parser.add_argument('--hidden_units', action="store", dest='hidden', default=500)
parser.add_argument('--amount_classes', action="store", dest='classes', default=102, help='amount of classes')
parser.add_argument('--epochs', action="store", dest='epochs',default=1)
parser.add_argument('--device', action="store", dest='device', choices={'cuda','cpu'},default='cpu')
results = parser.parse_args()


# 1. get data
# subfolders
data_dir = results.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
# Define your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(p=0.3),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
image_trainset = datasets.ImageFolder(train_dir, transform = train_transforms)
image_validset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
image_testset = datasets.ImageFolder(test_dir, transform = test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(image_trainset, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(image_validset, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(image_testset, batch_size = 64, shuffle = True)
# label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

'''

2a. BUILD THE NETWORK

'''

# get a pretrained model
if results.arch == 'densenet121':
    model = models.densenet121(pretrained=True)
else:
    model = models.vgg16(pretrained=True)

# freeze the parameters
for param in model.parameters():
    param.requires_grad = False

# create an own classifier
classifier = nn.Sequential(nn.Linear(1024,results.hidden),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(results.hidden,results.classes),
                           nn.LogSoftmax(dim=1))
# replace the model's classifier with the own one
model.classifier = classifier

# computing options
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = results.device
model.to(device)

'''

2b. TRAIN THE NETWORK


'''



print("train model")

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=results.learn_rate)

epochs = results.epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
# 4. Save the checkpoint

model.class_to_idx = image_trainset.class_to_idx
dict={}
dict['dropout']=0.2
dict['learn_rate']=results.learn_rate
dict['epochs']=results.epochs
dict['optimizer.state_dict()']=optimizer.state_dict()
dict['model']=model
dict['cat_to_name'] = cat_to_name
torch.save(dict, results.save_dir)



