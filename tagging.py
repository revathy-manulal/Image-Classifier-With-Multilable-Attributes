import torch
from torch import nn
from torch import optim
from torch.nn import functional
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import tag_data

PATH="/mnt/NTFS1/image-tagging"

df = tag_data.Tagging(PATH)

class TaggingDataset(Dataset):

    def __init__(self, dataset, dataset_path, transformer=None):

        super().__init__()

        self.dataset = dataset
        self.dataset_path = dataset_path
        self.transformer = transformer
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        item = self.dataset[index]
        img_path, category, attributes = item

        image = plt.imread(os.path.join(self.dataset_path, img_path))

        if self.transformer is not None:
            image = self.transformer(image)
        else:
            image = self.to_tensor(image)

        return image, category.astype(np.uint8), attributes

    def __len__(self):
        return len(self.dataset)


transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=(224, 224)),transforms.ToTensor()])

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


ds_train = TaggingDataset(df.train_imgs, PATH, transformer=transform)
ds_test  = TaggingDataset(df.test_imgs, PATH, transformer=transform)
ds_val   = TaggingDataset(df.val_imgs, PATH, transformer=transform)

val = next(iter(ds_train))

# print("The Image")
# print(type(val[0]))
# print(val[0].shape)


dl_train = data.DataLoader(ds_train, batch_size=512, shuffle=True, num_workers=4)
dl_test = data.DataLoader(ds_test, batch_size=512, shuffle=True, num_workers=4)
dl_val = data.DataLoader(ds_val, batch_size=512, shuffle=True, num_workers=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

resnet = models.resnet34(pretrained=True)
# print(resnet)
criterion = nn.BCEWithLogitsLoss()
resnet.avgpool = nn.AdaptiveAvgPool2d(1)

#removing the last layer
#list(resnet.children())[-1]

#turning out gradients for the resnet model
for param in resnet.parameters():
    param.requires_grad = False

from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(512, 1000)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(1000, 50))
                          ]))

resnet.classifier = classifier

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(resnet.classifier.parameters(), lr=0.003)

#move model to GPU if possible
resnet.to(device)
print(resnet)

epochs = 1
steps = 0
print_every = 5
running_loss = 0

for epoch in range(epochs):
    for images, categories, attributes in dl_train:
        steps += 1
        #images = images.view(images.shape[0], -1)
        images, categories, attributes = images.to(device), categories.to(device), attributes.to(device)
        #training_images = training_images.view(training_images.shape[0], -1)
        optimizer.zero_grad()

        attributes = attributes.type(torch.cuda.FloatTensor)

        logps = resnet(images)
        
        loss = criterion(logps, attributes)
        # loss.backward()
        optimizer.step()

        # running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            resnet.eval()
            with torch.no_grad():
                for images, categories, attributes in dl_val:
                    steps += 1
                    images, categories, attributes = images.to(device), categories.to(device), attributes.to(device)
                    attributes = attributes.type(torch.cuda.FloatTensor)
                    logps = resnet.forward(images)
                    batch_loss = criterion(logps, attributes)
                    # test_loss += batch_loss.item()

                    # Calculate accuracy
                    # ps = torch.exp(logps)
                    # top_p, top_class = ps.topk(1, dim=1)
                    # equals = top_class == labels.view(*top_class.shape)
                    # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print("Epoch {epoch+1}/{epochs}.. ",
                  "Train loss: {running_loss/print_every:.3f}.. ",
                  "Test loss: {test_loss/len(testloader):.3f}.. ",
                  "Test accuracy: {accuracy/len(testloader):.3f}")

            running_loss = 0
            resnet.train()


torch.save(resnet.state_dict(), PATH+'/resnet.pt')








