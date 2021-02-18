import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
#from workspace_utils import active_session
import json
from IPython.display import clear_output
import time
from PIL import Image
import pandas as pd

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model

    image = Image.open(image)
    
    if image.size[0]<image.size[1]:
        size = (256,int(256*image.size[1]/image.size[0]))
    else:
        size = (int(256*image.size[0]/image.size[1]),256)
        
    
    image = image.resize((size))

    #print(image.size)
    width = image.size[0]
    x_delta = (width-224)/2
    height = image.size[1]
    y_delta = (height-224)/2
    (left, upper, right, lower) = (x_delta,y_delta,width-x_delta,height-y_delta)
    image = image.crop((left, upper, right, lower))
    np_image = np.asarray(image)
    #Normieren durch Teilen durch 256
    np_image = np_image / 256
    #Standardisieren durch abziehen der MIttelwerte und Teilen durch die Standardabweichungen
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for color in range(np_image.shape[2]):
        np_image[:,:,color]-=means[color]
        np_image[:,:,color]/=stds[color]


    np_image = np_image.transpose(2,0,1)
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk, cat_to_name, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = torch.Tensor(image)
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        logps = model.forward(image.reshape(1,3,224,224))
        ps = torch.exp(logps)
        top_p, top_idx = ps.topk(topk, dim=1)
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_class = list(map(lambda a: idx_to_class[a.item()],top_idx[0]))
        top_name = list(map(lambda a: cat_to_name[a],top_class))
    return top_name, top_p