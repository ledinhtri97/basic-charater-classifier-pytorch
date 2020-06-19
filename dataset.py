from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "background"}

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CHARDATA(Dataset):
    """`CHARDATA <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, csv_file, root_dir, train=False, infer=False, transform=None):
        """
        Args:
            gt_file (string): Path to the groundtruth file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.chars_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.train = train
        self.infer = infer
        self.transform = transform

    def __len__(self):
        return len(self.chars_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.chars_frame.iloc[idx, 0])
        image = io.imread(img_name)
        char_id = 0
        
        if (self.train or not self.infer):
            char_name = self.chars_frame.iloc[idx, 1]
            for i, v in ALPHA_DICT.items():
                if char_name == v:
                    char_id = i
                    break
        else:
            char_id = img_name
        sample = {'image': image, 'char': char_id}

        if self.transform:
            sample = self.transform(sample)

        return sample