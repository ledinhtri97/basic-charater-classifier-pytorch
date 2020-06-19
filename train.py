import torch
from utils import trainloader
from model import charNet
from model import criterion
from model import optimizer
from model import device

PATHSAVE_MODEL = './char_net.pth'

def train():
	for epoch in range(10):
		print('epoch: %d' % epoch)
		running_loss = 0.0

		for i, data in enumerate(trainloader, 0):
			#get the inputs; data is a list of [inputs, labels]
			#print(data)
			inputs, labels = data['image'].to(device), data['char'].to(device)

			#zero the parameter gradients
			optimizer.zero_grad()

			#forward + backward + optimize
			outputs = charNet(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			#print statistics
			running_loss += loss.item()
			if i % 100 == 99:
				print('[%d, %5d] loss: %3f' % (epoch + 1, i + 1, running_loss / 100))
				running_loss = 0.0

	print('Finished Training')
	torch.save(charNet.state_dict(), PATHSAVE_MODEL)