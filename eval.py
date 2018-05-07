import random
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
from network import *

parser = argparse.ArgumentParser(description='Easy Implementation of InfoGAN')

# model hyper-parameters
parser.add_argument('--image_size', type=int, default=28) # 64 for CelebA
parser.add_argument('--z_dim', type=int, default=62) # 128 for CelebA

# InfoGAN parameters
parser.add_argument('--cc_dim', type=int, default=1)
parser.add_argument('--dc_dim', type=int, default=10)

# misc
parser.add_argument('--db', type=str, default='mnist')  # Model Tmp Save
parser.add_argument('--model_path', type=str, default='./models/generator-4.pkl')  # Model Tmp to Load
parser.add_argument('--sample_path', type=str, default='./results/eval')  # Results
parser.add_argument('--image_path', type=str, default='/Tmp/guiroysi/infoGAN/CelebA/128_crop')  # Training Image Directory
parser.add_argument('--sample_size', type=int, default=100)
parser.add_argument('--sample_step', type=int, default=100)
parser.add_argument('--manualSeed', type=int, help='manual seed')                        



##### Helper Function for Image Loading
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

##### Helper Function for GPU Training
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

##### Helper Function for Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# InfoGAN Function (Gaussian)
def gen_cc(n_size, dim):
    return torch.Tensor(np.random.randn(n_size, dim) * 0.5 + 0.0)

# InfoGAN Function (Multi-Nomial)
def gen_dc(n_size, dim):
    codes=[]
    code = np.zeros((n_size, dim))
    random_cate = np.random.randint(0, dim, n_size)
    code[range(n_size), random_cate] = 1
    codes.append(code)
    codes = np.concatenate(codes,1)
    return torch.Tensor(codes)

######################### Main Function
def main():


    # Pre-Settings
    cudnn.benchmark = True
    global args
    args = parser.parse_args()
    print(args)

    # Generating seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)


    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

        
    # Network
    generator = Generator(args.db, args.z_dim, args.cc_dim, args.dc_dim)
    # Loading previously trained models
    generator.load_state_dict(torch.load(args.model_path))


    if torch.cuda.is_available():
        generator.cuda()

    """Evaluate generator"""
    fixed_noise = to_variable(torch.Tensor(np.zeros((args.sample_size, args.z_dim))))  # For Testing



    # save the sampled images (10 Category(Discrete), 10 Continuous Code Generation : 10x10 Image Grid)
    tmp = np.zeros((args.sample_size, args.cc_dim))
    for k in range(10):
        tmp[k * 10:(k + 1) * 10, 0] = np.linspace(-5, 5, 10)
    cc = to_variable(torch.Tensor(tmp))
    tmp = np.zeros((args.sample_size, args.dc_dim))
    for k in range(10):
        tmp[k * 10:(k + 1) * 10, k] = 2
    dc = to_variable(torch.Tensor(tmp))

    fake_images = generator(torch.cat((fixed_noise, cc, dc), 1))
    torchvision.utils.save_image(denorm(fake_images.data),
                                 os.path.join(args.sample_path, 'generated.png'), nrow=10)


if __name__ == "__main__":

    main()
