import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from tqdm.notebook import tqdm
%matplotlib inline

//---------------- Use the block of code below if you import data image straight on Kaggle
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
//-----------------

DATA_DIR = '../input/10-monkey-species'

TRAIN_DIR = DATA_DIR + '/training/training'                           
VAL_DIR = DATA_DIR + '/validation/validation'

labels_name= ['mantled_howler',
              'patas_monkey',
            'bald_uakari',
            'japanese_macaque',
            'pygmy_marmoset',
            'white_headed_capuchin',
            'silvery_marmoset',
            'common_squirrel_monkey',
            'black_headed_night_monkey',
            'nilgiri_langur']
    
transform = transforms.Compose ([ transforms.Resize(size=(256,256) , interpolation=2),transforms.ToTensor(),])

train_dataset = ImageFolder ( TRAIN_DIR , transform=transform )
val_dataset = ImageFolder ( VAL_DIR , transform=transform )

// print(train_dataset.classes)
//len(train_dataset)

def show_example(img, label):
    print('Label: ', train_dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))

show_example(*train_dataset[12])
show_example(*val_dataset[123])

random_seed = 40
torch.manual_seed(random_seed);
batch_size = 5

# Set aside some data for testing

test_size = 200

train_ds, test_ds = random_split(train_dataset, [len(train_dataset)-test_size, test_size])
len(train_ds), len(test_ds)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size, num_workers=4, pin_memory=True)


for images,labels in train_loader:
    print('images.shape:', images.shape)
    fig, ax = plt.subplots(figsize=(32, 16))
    plt.axis('on')
    ax.set_xticks([]); ax.set_yticks([])
    plt.imshow(make_grid(images, nrow=8).permute((1, 2, 0)))
    break





