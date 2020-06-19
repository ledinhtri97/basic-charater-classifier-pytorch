import torch
import torchvision
import os
from shutil import copyfile

from utils import testloader
from dataset import ALPHA_DICT
from model import charNet
from model import device

PATHSAVE_MODEL = './char_net.pth'
ROOT_TEST = './data'
ROOT_PREDICT = './predicted'

def test():
	dataiter = iter(testloader)
	charNet.load_state_dict(torch.load(PATHSAVE_MODEL))
	
	class_correct = list(0. for i in range(len(ALPHA_DICT)))
	class_total = list(0. for i in range(len(ALPHA_DICT)))

	with torch.no_grad():
		for data in testloader:
			images, path_images = data['image'].to(device), data['char']
			outputs = charNet(images)

			_, predicted = torch.max(outputs.data, 1)
			
			for i in range(len(path_images)):
				char_pre = predicted[i].item()
				print('%20s : %5s' % (path_images[i], ALPHA_DICT[char_pre]))
				#copyfile(src, dst)
	

