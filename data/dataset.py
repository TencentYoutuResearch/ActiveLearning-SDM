from torchvision.datasets import VisionDataset
import torch
from PIL import Image
import os
import sys
import numpy as np

# load different dataset
def get_data(name,path,transform=None,tr_or_te='train',n_views=1):

    assert name in ['OfficeHome', 'DomainNet', 'Office31']

    if name == 'OfficeHome':
        return ImageList(path+'.txt',transform=transform,n_views=n_views)
    elif name == 'DomainNet':
        return ImageList(path+'_'+tr_or_te+'.txt',transform=transform,n_views=n_views)
    elif name == 'Office31':
        return ImageList(path+'.txt',transform=transform,n_views=n_views)

# PIL loader
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# active learning should implement add_item function and remove_item function
class ImageList(VisionDataset):
    def __init__(self, root, transform=None, contrastive_transform=None, n_views=1):
        super(ImageList, self).__init__(root, transform=transform)

        # self.samples = np.loadtxt(root, dtype=np.unicode_, delimiter=' ')
        self.samples = np.loadtxt(root, dtype=np.dtype((np.unicode_, 1000)), delimiter=' ')
        self.loader = pil_loader
        self.contrastive_transform = contrastive_transform
        self.n_views = n_views

    def __getitem__(self, index):

        path, target = self.samples[index]
        target = int(target)

        sample = self.loader(path)

        if self.transform is not None:
            if self.n_views == 1:
                sample = self.transform(sample)
            else:
                sample = [self.transform(sample) for i in range(self.n_views)]
                sample = torch.stack(sample, dim=0)
                # sample = torch.cat(sample, dim=1)
                # sample = self.transform(sample)

        return sample, target, path, index

    def __len__(self):
        return len(self.samples)

    def add_item(self, addition):
        self.samples = np.concatenate((self.samples, addition), axis=0)
        return self.samples

    def remove_item(self, reduced):
        reduced = reduced.astype('int64')
        self.samples = np.delete(self.samples, reduced, axis=0)
        return self.samples
