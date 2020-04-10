# -*- coding: utf-8 -*-
# License: BSD
# Based on the tutorial by Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os, sys
import copy
import pickle

FEATS = 7
if len(sys.argv) > 1:
    input_dir = sys.argv[1]
    print(input_dir)
    if input_dir[-1] != '/':
        input_dir = input_dir + '/'
    try:
## Bit hacky, but it grabs the last integer, expecting that to be the ros time
        ROS_START = input_dir.split('/')[-2].split('_')[-1]
        ROS_START = int(ROS_START)
    except:
        import pdb
        pdb.set_trace()
        print('ROS time not found')
        ROS_START = 0
else:
    input_dir = 'data/test_pngs/'
    ROS_START = 0
data_transforms = {
    'predict': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
dataset = {'predict':datasets.ImageFolder(input_dir,data_transforms['predict'])}
dataloader = {'predict':torch.utils.data.DataLoader(dataset['predict'],batch_size = 4,shuffle=False,num_workers=4)}
"""
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9,1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
## Test it
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

#dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('loading model:')
loaded_model = models.resnet18(pretrained=True)
num_ftrs = loaded_model.fc.in_features
loaded_model.fc = nn.Linear(num_ftrs, FEATS)
loaded_model = loaded_model.to(device)

## Load weights
loaded_model.load_state_dict(torch.load('/home/ammon/Documents/Scripts/SegmentSound/SongClassifier/song_model.pt'))

print('Prep for inference')
loaded_model.eval()

output = {}

import pdb
#pdb.set_trace()

true_i = 0

class_names = ['burble','chuck','human','noise','overlap','song','t']
with torch.no_grad():
    for inputs,labels in dataloader['predict']:
        inputs = inputs.to(device)
        #labels = labels.to(device)
        outputs = loaded_model(inputs)
        maxes, predictions = torch.max(outputs,1) 
        for j in range(inputs.size()[0]):
## Append the img names, predictions, and confidence
            #pdb.set_trace()
            img_name = dataset['predict'].imgs[true_i]
            output[true_i] = {
                'img_name' :img_name,'prediction':predictions[j].cpu().numpy(),
                'class_name':class_names[predictions[j]],'max':maxes[j].cpu().numpy()}
            true_i += 1

print(outputs)
print('0:',class_names[0])
print('1:',class_names[1])

print('writing to file:')
with open('predictions2.pkl','wb') as outfile:
    pickle.dump(output,outfile)

## Take a output dict and convert it into a nice line
def parse_line(out_line):
    img_name = out_line['img_name'][0]
    trim_name = img_name.split('/')[-1].split('.')[0]
    times = trim_name.split('_')[1] 
    start_time,stop_time = times.split('-') 
    event = out_line['class_name']
    ros_start = int(start_time) + ROS_START
    ros_stop = int(stop_time) + ROS_START
    out_line = ','.join([start_time,event,stop_time,str(ros_start),str(ros_stop)])
    return out_line

print('Making summary')
save_classes=['song','burble']
with open(input_dir + 'summary.txt','w') as out_file:
    for l in range(len(output)):
        line = output[l]
        if line['class_name'] in save_classes:
           out_file.write(parse_line(line) + '\n')

print('all done!')
exit()

