import torch
import torchvision
import torchvision.transforms as trf
import matplotlib.pyplot as plt
import numpy as np
from dataset import CHARDATA, ALPHA_DICT
from transforms import Rescale, ToTensor, Normalize

transform = trf.Compose(
	[	
		Rescale((28,28)),
		Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTensor()
	])

trainset = CHARDATA(csv_file='data/groundtruth.csv', root_dir='data', 
	train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

validset = CHARDATA(csv_file='data/groundtruth.csv', root_dir='data', transform=transform)
validloader = torch.utils.data.DataLoader(validset, batch_size=4, shuffle=False, num_workers=2)

testset = CHARDATA(csv_file='data/testset.csv', root_dir='data', 
	infer=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

def showimage(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()

def test_this():
	dataiter = iter(trainloader)
	images, labels = dataiter.next()
	#show images
	showimage(torchvision.utils.make_grid(images))
	#print labels
	print(' '.join('%5s' % ALPHA_DICT[labels[j]] for j in range(4)))
