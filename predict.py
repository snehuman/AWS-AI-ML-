import torch
from torchvision import datasets, transforms
import helper
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from torchvision import models
from torch import nn, optim
import torch.nn as nn
import argparse
import json

parser = argparse.ArgumentParser(description = " predict the flower name ")

parser.add_argument('data_dir', type = str, help = "training")
parser.add_argument('--top_k', type=int, default=5, help="Return top K predictions", dest="top_k")
parser.add_argument('--save_dir', type = str, help = "save the checkpoint" , dest="save_dir" )
parser.add_argument('--category_names', type=str, dest = 'category_names', default='cat_to_name.json' )
parser.add_argument('--gpu', help = 'usage of gpu', action = 'store_true' )
parser.add_argument('--learning_rate', type = float, help = "learning rate", action="store", dest="learning_rate", default=0.001)
parser.add_argument('--hidden_layer', type = int, help = "number of hidden layer", action = 'store')
parser.add_argument('--epochs', type = int, help = "number of epochs", default = 5)
parser.add_argument('--arch', type=str, default = 'resnet18', help="Choose the model architecture", action="store")



args = parser.parse_args()

data_dir = args.data_dir
top_k = args.top_k
save_dir = args.save_dir
category_names = args.category_names
gpu = args.gpu
learning_rate = args.learning_rate
hidden_layer = args.hidden_layer
epochs = args.epochs
arch = args.arch

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform = test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle = True)



# for choosing resnet 18 , this is deafault

if arch == 'resnet50':
  model = models.resnet50(pretrained=True)
  for param in model.parameters():
      param.requires_grad = False

else:
  model = models.resnet18(pretrained=True)
  for param in model.parameters():
      param.requires_grad = False


model.fc = nn.Sequential(nn.Linear(512, 256),
                          nn.ReLU(),
                          nn.Dropout(0.3),
                          nn.Linear(256, 152),
                          nn.ReLU(),
                          nn.Dropout(0.3),
                          nn.Linear(152, 102),
                          nn.LogSoftmax(dim=1))

for param in model.fc.parameters():
  param.requires_grad = True


criterion = nn.NLLLoss()

optimizer = optim.Adam(model.fc.parameters(), lr = 0.001)

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)


epochs = 5
train_losses, valid_losses = [], []

for e in range(epochs):
  training_loss = 0
  model.train()

  for images, labels in trainloader:

    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()

    log_ps = model(images)
    loss = criterion(log_ps, labels)
    loss.backward()
    optimizer.step()

    training_loss += loss.item()

  else:
    valid_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
      for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)

        log_ps = model.forward(images)
        valid_loss += criterion(log_ps, labels).item()

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        train_losses.append(training_loss / len(trainloader))
        valid_losses.append(valid_loss / len(validloader))



model.state_dict().keys()

class_to_index = train_data.class_to_idx
model.class_to_index = class_to_index

torch.save(model.state_dict(), 'checkpoint.pth')
state_dict = torch.load('checkpoint.pth', map_location = torch.device('cpu'))
model.load_state_dict(state_dict)

model.class_to_index = train_data.class_to_idx

checkpoint = {
        'model': model.fc,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_index,
        'optimizer_state_dict': optimizer.state_dict(),

}

torch.save(checkpoint, 'checkpoint.pth')


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(256, 152),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(152, 102),
                                nn.LogSoftmax(dim=1))

    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_index = checkpoint['class_to_idx']
    model.to(device)
    return model, optimizer

model, optimizer = load_checkpoint('checkpoint.pth')

def process_image(image_path):
    img = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = preprocess(img)

    return img


def predict(image_path, model, topk=5):

    img = process_image(image_path)

    img = img.unsqueeze(0)

    img = img.to(device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        log_ps = model(img)
        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(topk, dim=1)

    idx_to_class = {v: k for k, v in model.class_to_index.items()}
    top_class_names = [idx_to_class[i.item()] for i in top_class[0]]

    return top_p[0].cpu().numpy(), top_class_names


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(image, torch.Tensor):
        image = image.numpy()

    image = image.transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def check(image_path, model, topk=5):

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict(image_path, model, topk=topk)

    flower_names = [cat_to_name[str(cls)] for cls in classes]

    img = process_image(image_path)

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), nrows=2)

    imshow(img, ax=ax1)
    ax1.set_title(flower_names[0])

    y_pos = np.arange(len(flower_names))
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(flower_names)
    ax2.invert_yaxis()

    plt.show()

image_path = '/content/flowers/test/34/image_06929.jpg'
check(image_path, model, topk=5)

