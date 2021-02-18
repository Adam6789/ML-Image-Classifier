import numpy as np
import argparse
import torch
import json
import utils

parser = argparse.ArgumentParser(
    description='This is a program to predict the class of an image',
)
# positional arguments
parser.add_argument( action="store", dest='image_path')
parser.add_argument( '--load_dir', action="store", dest='load_dir',default='final_model.pth')
# optional arguments
parser.add_argument('--top_k', action="store", type=int, dest='top_k', default=1)
parser.add_argument('--category_names', action="store", dest='category_names')
parser.add_argument('--device', action="store", dest='device', choices={'cuda','cpu'}, default='cpu')
results = parser.parse_args()
device = results.device

model=torch.load(results.load_dir, map_location=device)['model']
model.to(device)

if results.category_names is not None:
    with open(results.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)  



top_name, top_p = utils.predict(results.image_path, model, results.top_k, cat_to_name, device)

print("top category resp. categories:", top_name)
print("top probability resp. probabilities", top_p[0])





