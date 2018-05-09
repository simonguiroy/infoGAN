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
from infogan_multiple_dc import *
from utils import *

parser = argparse.ArgumentParser(description='InfoGAN')

parser.add_argument('--model_path', type=str, default='./models_multiple_dc/generator-1.pkl') 
parser.add_argument('--sample_path', type=str, default='./results/eval_multiple')  
parser.add_argument('--image_path', type=str, default='/Tmp/guiroysi/infoGAN/CelebA/128_crop') 
parser.add_argument('--sample_size', type=int, default=100)
parser.add_argument('--sample_step', type=int, default=100)
parser.add_argument('--manualSeed', type=int, help='manual seed') 

# model hyper-parameters
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--z_dim', type=int, default=128) 

# InfoGAN parameters
parser.add_argument('--cc_dim', type=int, default=1)
parser.add_argument('--dc_dim', type=int, default=10)
parser.add_argument('--dc_num', type=int, default=1)

parser.add_argument('--dc_vary', type=int, default=1, help='the dc variable to vary')




def main():


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
    generator = Generator(args.z_dim, args.cc_dim, args.dc_dim, args.dc_num)
    # Loading previously trained models
    generator.load_state_dict(torch.load(args.model_path))


    if torch.cuda.is_available():
        generator.cuda()

    #fixed_noise = to_variable(torch.Tensor(np.zeros((args.sample_size, args.z_dim))))  # For Testing
    fixed_noise = to_variable(torch.FloatTensor(args.sample_size, args.z_dim, 1, 1).normal_(0, 1))



    #generating evaluation samples
    tmp = np.zeros((args.sample_size, args.cc_dim))

    for k in range(10):
        tmp[k * 10:(k + 1) * 10, 0] = np.linspace(-5, 5, 10)
    cc = to_variable(torch.Tensor(tmp))
    
    temps = []
    for id_dc in range(args.dc_num):
        tmp = np.zeros((args.sample_size, args.dc_dim))
        if id_dc == args.dc_vary:
            for k in range(10):
                tmp[k * 10:(k + 1) * 10, k] = 1    
        else:
            tmp[:,int(args.dc_dim/2)] = 1
        temps.append(tmp)
    temps = np.concatenate(temps,1)
    
    dc = to_variable(torch.Tensor(temps))

    fake_images = generator(torch.cat((fixed_noise, cc, dc), 1))
    torchvision.utils.save_image(fake_images.data,
                                 os.path.join(args.sample_path, 'generated-%d.png' % args.dc_vary), nrow=10)


if __name__ == "__main__":

    main()
