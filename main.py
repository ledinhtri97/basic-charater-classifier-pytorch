from train import train
from valid import valid
from test import test
import argparse
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--do", required=True,
	help="DO WHAT train/test?")
args = vars(ap.parse_args())

if __name__ == '__main__':
	if args["do"] == 'train' or args["do"] == 'TRAIN':
		train()
	elif args["do"] == 'valid' or args["do"] == 'valid':
		valid()
	elif args["do"] == 'test' or args["do"] == 'TEST':
		test()