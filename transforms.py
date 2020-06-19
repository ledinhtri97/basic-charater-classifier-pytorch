import torch
from skimage import transform
import numpy as np

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, char_name = sample['image'], sample['char']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'char': char_name}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, char_name = sample['image'], sample['char']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'char': char_name}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
        Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
        channels, this transform will normalize each channel of the input
        ``torch.*Tensor`` i.e.,
        ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

        .. note::
            This transform acts out of place, i.e., it does not mutate the input tensor.

        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation in-place.
	"""
    
    def __init__(self, mean, std):
        
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            image tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        image, char_name = sample['image'], sample['char']
        image = image.astype(np.float32)
        image -= self.mean
        image /= self.std
        return {'image': image, 'char': char_name}


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
