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

parser = argparse.ArgumentParser(description = " Training the neural network")

parser.add_argument('data_dir', type = str, help = "training")
parser.add_argument('--save_dir', type = str, help = "save the checkpoint" ,action="store", dest="save_dir" )
parser.add_argument('--learning_rate', type = float, help = "learning rate", action="store", dest="learning_rate", default=0.001)
parser.add_argument('--hidden_layer', type = int, help = "number of hidden layer", action = 'store')
parser.add_argument('--epochs', type = int, help = "number of epochs", default = 5)
parser.add_argument('--gpu', help = 'usage of gpu', action = 'store_true' )
parser.add_argument('--arch', type=str, default = 'resnet18', help="Choose the model architecture", action="store")

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
learning_rate = args.learning_rate
hidden_layer = args.hidden_layer
epochs = args.epochs
gpu = args.gpu
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

    print(f"Epoch: {e+1}/{epochs}.. "
          f"Training Loss: {training_loss / len(trainloader):.3f}.. "
          f"valid Loss: {valid_loss / len(validloader):.3f}.. "
          f"Accuracy: {accuracy / len(validloader):.3f}")




def saveCheckpoint(model, save_path):
  torch.save(model.state_dict(), 'checkpoint.pth')
  model.load_state_dict(torch.load('checkpoint.pth', map_location = torch.device('cpu')))

  model.class_to_index = train_data.class_to_idx

  checkpoint = {
          'model': model.fc,
          'state_dict': model.state_dict(),
          'class_to_idx': model.class_to_index,
          'optimizer_state_dict': optimizer.state_dict(),
          }

if args.save_dir:
  saveCheckpoint(model, save_dir + '/checkpoint.pth')
else:
  saveCheckpoint(model, 'checkpoint.pth')
