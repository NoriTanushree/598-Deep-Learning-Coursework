
# coding: utf-8

# ### IE 534 : HW-4
# #### Residual Networks
# _Tanushree Nori_

# -  Build the Residual Network specified in the HW and achieve at least 60% test accuracy.
#     In the homework, you should define your “Basic Block” as shown in Figure 2. For each
#     weight layer, it should contain 3 × 3 filters for a specific number of input channels and
#     output channels. The output of a sequence of ResNet basic blocks goes through a max
#     pooling layer with your own choice of filter size, and then goes to a fully-connected
#     layer. The hyperparameter specification for each component is given in Figure 1. Note
#     that the notation follows the notation in He et al. (2015)
# 
# 
# 
# -  Fine-tune a pre-trained ResNet-18 model and achieve at least 70% test accuracy.

# In[2]:


#  Import libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from time import time


# In[3]:


class Block(nn.Module):
    def __init__(self, input_c, output_c, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size = 3, padding = 1, stride = stride, bias = False)
        self.bn1 = nn.BatchNorm2d(output_c)
        self.conv2 = nn.Conv2d(output_c, output_c, kernel_size = 3, padding = 1, stride = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_c)
        
        self.shortcut = nn.Sequential()
        
        #if the input num of channels is not equal to output, output is resized
        if input_c != output_c:
            self.shortcut = nn.Sequential(
            nn.Conv2d(input_c, output_c, kernel_size = 1, stride = stride, bias = False),
            nn.BatchNorm2d(output_c)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return out 
  


# In[4]:



class ResNet(nn.Module):
    def __init__(self, basicblock, num_blocks, classes = 100):
        super(ResNet, self).__init__()
        self.input_c = 32
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(p = 0.2)
        self.layer1 = self._make_layer(basicblock, 32, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(basicblock, 64, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(basicblock, 128, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(basicblock, 256, num_blocks[3], stride = 2)
        self.fc = nn.Linear(256 * 2 * 2, classes)
        
    # Form a resnet block as in the HW description
    def _make_layer(self, basicblock, output_c, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(basicblock(self.input_c, output_c, stride))
            self.input_c = output_c
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.max_pool2d(out, kernel_size = 3, stride = 2, padding = 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
        
        
def ResNet598():
    return ResNet(Block, [2,4,4,2])


# ### Part 1

# <font color = blue> Load CIFAR 100 and apply data augmentation techniques (horizontal flip and random crop) </font>
# 
# <font color = blue> Drop out is set to 0.2 (see above in resnet defintion) </font>

# In[11]:


transform_training = transforms.Compose([
								transforms.RandomCrop(32, padding=4),
								transforms.RandomHorizontalFlip(),
								transforms.ToTensor()])
								
transform_test = transforms.Compose([transforms.ToTensor()])
								
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_training)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)


# <font color = blue> Define learning parameter to be 10^(-3) </font>
# 
# 
# <font color=blue> We use cross entropy loss function and the Adam optimizer </font>

# In[12]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ResNet598()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)


# <font color=blue> Training is done for 65 epochs for this part </font>

# In[13]:


start = time()
for epoch in range(65):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0


end = time()
print('Finished Training, it took {} seconds'.format(end - start))

#Getting the accuracy of the model

correct = 0
total = 0

with torch.no_grad():
    
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total +=labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy of the Resnet is {} %".format(100 * correct / total))


# **Final accuracy with the pre-trained model is 61.02%**

# ### Part 2 

# - Load CIFAR 100 and upsample it to 224x224 to match the input size from the pre-trained model 
# 
# 
# - The data augmentation techniques applied are the same as above except for the additional resize parameter 

# In[5]:



transform_training_pretrain = transforms.Compose([
                                transforms.Resize(size=(224,224)),
								transforms.RandomCrop(224, padding=4),
								transforms.RandomHorizontalFlip(),
								transforms.ToTensor()])
								
transform_test_pretrain = transforms.Compose([transforms.Resize(size = (224,224)),
                                        transforms.ToTensor()])
								
trainset_pretrain = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_training_pretrain)

trainloader_pretrain = torch.utils.data.DataLoader(trainset_pretrain, batch_size=100,
                                          shuffle=True, num_workers=2)

testset_pretrain = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test_pretrain)
testloader_pretrain = torch.utils.data.DataLoader(testset_pretrain, batch_size=100,
                                         shuffle=False, num_workers=2)


# - Loading the pre-trained model and running the optimizer on that. We fine tune the pretrained model

# In[6]:



model_ft = torchvision.models.resnet18(pretrained=True)

for param in model_ft.parameters():
    param.requires_grad = True
    
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 100)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)


# - Training is done for 10 epochs 
# - Learning parameter is again 10^(-3) 
# - Adam optimizer is used 

# In[8]:


#  Model training and output
start = time()
for epoch in range(10):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader_pretrain, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0


end = time()
print('Finished Training, it took {} seconds'.format(end - start))


# In[15]:


#Getting the accuracy of the model

correct = 0
total = 0

with torch.no_grad():
    
    for data in testloader_pretrain:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model_ft(images)
        _, predicted = torch.max(outputs.data, 1)
        total +=labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy of the pre-trained network is {} %".format(100 * correct / total))


# **Final accuracy with the pre-trained model is 72.39%**
