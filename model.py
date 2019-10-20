import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.pool = nn.MaxPool2d(2,2)
		self.conv1 = nn.Conv2d(1,64,5)
		self.conv2 = nn.Conv2d(64,128,3)
		self.conv3 = nn.Conv2d(128,256,3)
		self.conv4 = nn.Conv2d(256,512,3)
		self.conv5 = nn.Conv2d(512,1024,3)
		self.fc1 = nn.Linear(25600, 4096)
		self.fc2 = nn.Linear(4096, 1024)
		self.fc3 = nn.Linear(1024, 136)		

	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = self.pool(F.relu(self.conv4(x)))
		x = self.pool(F.relu(self.conv5(x)))
		x = x.view(x.size(0),-1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
		
