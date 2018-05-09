import argparse
import os
import torch
import torchvision
import numpy as np
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from PIL import Image

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None):  # Initializes image paths and preprocessing module.
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform

    def __getitem__(self, index):  # Reads an image from a file and preprocesses it and returns.
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):  # Returns the total number of image files.
        return len(self.image_paths)

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

    
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def gen_cc(n_size, dim):
    return torch.Tensor(np.random.randn(n_size, dim) * 0.5 + 0.0)

def gen_dc(n_size, dim):
    codes=[]
    code = np.zeros((n_size, dim))
    random_cate = np.random.randint(0, dim, n_size)
    code[range(n_size), random_cate] = 1
    codes.append(code)
    codes = np.concatenate(codes,1)
    return torch.Tensor(codes)
    
    
def gen_dc_multiple(n_size, dim, num):
    codes=[]
    for i in range(num):
        code = np.zeros((n_size, dim))
        random_cate = np.random.randint(0, dim, n_size)
        code[range(n_size), random_cate] = 1
        codes.append(code)
    codes = np.concatenate(codes,1)
    return torch.Tensor(codes)