# Trained a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset. The convolution network used (A) dropout, (B) trained with RMSprop or ADAM, and (C) data augmentation


#module load bwpy/2.0.0-pre1
#module load cudatoolkit
# coding: utf-8
from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from time import time

def conv_block_gen(in_f, out_f, size, *args, **kwargs):
        return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.LayerNorm(out_f,size,size),
        nn.leaky_relu()
       )
		

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		
		
		self.encoder = nn.Sequential
		(
			conv_block_gen(3,196, size =32, kernel_size = 3, stride = 1, padding = 1),
			conv_block_gen(196,196, size =16,kernel_size = 3, stride = 2, padding = 1 ),
			conv_block_gen(196,196, size =16, kernel_size = 3, stride = 1, padding = 1),
			conv_block_gen(196,196, size =8, kernel_size = 3, stride = 2, padding = 1),
			conv_block_gen(196,196, size =8, kernel_size = 3, stride = 1, padding = 1),
			conv_block_gen(196,196, size =8, kernel_size = 3, stride = 1, padding = 1),
			conv_block_gen(196,196, size =8, kernel_size = 3, stride = 1, padding = 1),
			conv_block_gen(196,196, size =4, kernel_size = 3, stride = 2, padding = 1)
			nn.MaxPool2d(kernel_size=4, stride=4, padding=0)	
		)
		
		 self.fc1 = nn.Linear(196, 10)
         self.fc2 = nn.Linear(196,1)

		

	def forward(self, x):
		
		x = self.encoder(x)
		
		x = x.view(x.size(0),-1)
		
		x = self.fc1(x)

		return x


transform_train = transforms.Compose([
		transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
		transforms.ColorJitter(
		brightness=0.1*torch.randn(1),
		contrast=0.1*torch.randn(1),
		saturation=0.1*torch.randn(1),
		hue=0.1*torch.randn(1)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

transform_test = transforms.Compose([
transforms.CenterCrop(32),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

model = Discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model_pt.load_state_dict(torch.load('epochs_10.ckpt'))

def train(epoch):
	#iterating over output of dataiter(batches), running model and optimizing
	start = time()
	for epoch in range(100):
		
		running_loss = 0.0

		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)
			# zero the parameter gradients
			optimizer.zero_grad()
			
			# forward + backward + optimize
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()	
			optimizer.step()
			
			# print statistics
			running_loss += loss.item()
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
                  (e + 1, batch_idx + 1, running_loss / 500))
            running_loss = 0.0

	end = time()
	finished_statement = ('Finished Training, it took {} seconds'.format(end - start))
			

correct = 0
total = 0

with torch.no_grad(): 
		
		for data in testloader:
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total +=labels.size(0)
			correct += (predicted == labels).sum().item()

	accuracy = ("Accuracy of this NN on test data is {} %".format(100 * correct / total))
	


# Generator bit

# Generate noise variables of 100 * 100
np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0,1,(100,n_z))
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()
	
def conv_block_gen(in_f, out_f, *args, **kwargs):
        return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.relu()
       )

def convtrans_block_gen(in_f, out_f, *args, **kwargs):
        return nn.Sequential(
        nn.ConvTranspose2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.relu()
       )
			   

class Generator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		
		self.fc1 = nn.Linear(100, 196*4*4)
		
		self.encoder = nn.Sequential
		(   
			convtrans_block_gen(196,196, kernel_size = 4, stride = 2, padding = 1 ),
			conv_block_gen(196,196, kernel_size = 3, stride = 1, padding = 1),
			conv_block_gen(196,196, kernel_size = 3, stride = 1, padding = 1),
			conv_block_gen(196,196, kernel_size = 3, stride = 1, padding = 1),
			convtrans_block_gen(196,196, kernel_size = 4, stride = 2, padding = 1),
			conv_block_gen(196,196, kernel_size = 3, stride = 1, padding = 1),
			convtrans_block_gen(196,196, kernel_size = 4, stride = 2, padding = 1),
			nn.Conv2d(196,196, kernel_size = 3, stride = 1, padding = 1)	
			nn.Tanh()
		)
		
	def forward(self, x):
		x = self.fc1(x)
		x = x.view(196,4,4)
		x = self.encoder(x)

		return x
			
		
