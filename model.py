import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NUM_CLASS = 32

class CharNet(nn.Module):
	def __init__(self):
		super(CharNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, 5) #outchannel 32
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(32, 64, 5) #inchannel 32 equal previous conv outchannel
		self.fc1 = nn.Linear(64*4*4, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 32) #_, NUM_CLASS

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		#print(x.size())
		x = x.view(-1, 64*4*4)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

# MODEL
charNet = CharNet()

#TRAINING ON GPU OPTION
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on ... ', device)
charNet.to(device)

# LOSS AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(charNet.parameters(), lr=0.001, momentum=0.9)



