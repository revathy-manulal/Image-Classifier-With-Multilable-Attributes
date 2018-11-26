import torch
from torch import nn
from torch import optim
from torch.nn import functional
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.dataset import Dataset
from collections import OrderedDict
from os import listdir

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import tag_data
#import helper

PATH="/home/aman/img_tagging"

class TaggingDataset(Dataset):

    def __init__(self, dataset, dataset_path, transformer=None):

        super().__init__()

        self.dataset = dataset
        self.dataset_path = dataset_path
        self.transformer = transformer
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        img_path = self.dataset[index]

        image = plt.imread(os.path.join(self.dataset_path+ '/prediction/images', str(img_path)))
        # print(image)
        
        if self.transformer is not None:
            image = self.transformer(image)
        else:
            image = self.to_tensor(image)

        return image

    def __len__(self):
        return len(self.dataset)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

onlyfiles = [f for f in listdir(PATH+ '/prediction/images')]
print(onlyfiles)
# resnet34
resnet = models.resnet50(pretrained=True)

resnet.avgpool = nn.AdaptiveAvgPool2d(1)

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(512, 1000)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1000, 50))
                          ]))

resnet.classifier = classifier

data_dir = PATH


test_transforms=transforms.Compose([transforms.ToPILImage(),
    transforms.Resize(255),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
    ])
#test_data = datasets.ImageFolder(data_dir + '/prediction', transform=test_transforms)
test_data = TaggingDataset(onlyfiles , PATH, transformer=test_transforms)

images= next(iter(test_data))
# print(images)
# print(images.shape)

dl_train = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False)


resnet.load_state_dict(torch.load(PATH+'/resnet.pt'))
resnet.to(device)
#print(testloader)


image_to_category = []
with open("/home/aman/img_tagging/lib/category_list.txt") as f:

    for line in f:
        words = line.split()
        att = " ".join(words)
        image_to_category.append(att)

# print(image_to_category)
# print(len(image_to_category))

for images in dl_train:
    
    images = images.to(device)
    resnet.eval()
    logits = resnet.forward(images)
    prob = functional.sigmoid(logits)
    # print(prob)
    # print('patterns')
    out_put = prob.data.cpu().numpy()[0]
    categories = []
    for value in range(len(out_put)):
        if out_put[value] > 0.985:
            categories.append(image_to_category[value])

    print(categories)






	
