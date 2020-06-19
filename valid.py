import torch
import torchvision
from utils import validloader
from dataset import ALPHA_DICT

from model import charNet
from model import device

PATHSAVE_MODEL = './char_net.pth'

def valid():
	charNet.load_state_dict(torch.load(PATHSAVE_MODEL))
	correct = 0
	total = 0
	class_correct = list(0. for i in range(len(ALPHA_DICT)))
	class_total = list(0. for i in range(len(ALPHA_DICT)))

	with torch.no_grad():
		for data in validloader:
			images, labels = data['image'].to(device), data['char'].to(device)
			outputs = charNet(images)

			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

			c = (predicted == labels).squeeze()
			for i in range(len(labels)):
				label = labels[i].item()
				
				if (c[i].item()):
					class_correct[label] += 1
				class_total[label] += 1

	print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))

	print('More detail:')
	for i in range(len(ALPHA_DICT)):
		print('Accuracy of %5s : %2d %%' % (
			ALPHA_DICT[i], 100 * class_correct[i] / (class_total[i]+1)))
	

