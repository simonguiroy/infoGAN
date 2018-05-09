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

#paths and log params
parser.add_argument('--model_path', type=str, default='./models_multiple_dc')  # path to save trained models, using multiple discrete codes
parser.add_argument('--image_path', type=str, default='/Tmp/guiroysi/CelebA/128_crop')  # path of data
parser.add_argument('--sample_size', type=int, default=100)
parser.add_argument('--log_step', type=int, default=50)
parser.add_argument('--sample_step', type=int, default=200)

# if resuming training
parser.add_argument('--path_G', default='', help="path to Generator (to continue training)")
parser.add_argument('--path_D', default='', help="path to Discriminator (to continue training)")
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch if continuing training')

# image size and noise dimensions
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--z_dim', type=int, default=128)

#general training hyperparameters
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--lrD', type=float, default=0.0002)
parser.add_argument('--lrG', type=float, default=0.001) 
parser.add_argument('--beta1', type=float, default=0.5) 
parser.add_argument('--beta2', type=float, default=0.999) 

#dimensions of latent codes
parser.add_argument('--cc_dim', type=int, default=1)
parser.add_argument('--dc_num', type=int, default=1, help='number of discrete codes')
parser.add_argument('--dc_dim', type=int, default=10)
parser.add_argument('--only_dc', action='store_true', help='only trains discrete code')





def main():
    cudnn.benchmark = True
    global args
    args = parser.parse_args()
    print(args)

    #normalizing data
    transform = transforms.Compose([
        transforms.Scale((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    #creating dataset and data loader
    dataset = ImageFolder(args.image_path, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)


    # Networks
    generator = Generator(args.z_dim, args.cc_dim, args.dc_dim, args.dc_num)
    discriminator = Discriminator(args.cc_dim, args.dc_dim, args.dc_num)
    
    # Loading previously trained models, if specified, to continue training
    if args.path_G != '':
        generator.load_state_dict(torch.load(args.path_G))
    if args.path_D != '':
        discriminator.load_state_dict(torch.load(args.path_D))
    
    # Optimizers
    optim_g = optim.Adam(generator.parameters(), args.lrD, [args.beta1, args.beta2])
    optim_d = optim.Adam(discriminator.parameters(), args.lrG, [args.beta1, args.beta2])

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()


    total_num_steps = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, images in enumerate(data_loader):
            # Training discriminator
            images = to_variable(images)

            batch_size = images.size(0)
            noise = to_variable(torch.randn(batch_size, args.z_dim))

            cc = to_variable(gen_cc(batch_size, args.cc_dim))
            dc = to_variable(gen_dc_multiple(batch_size, args.dc_dim, args.dc_num))

            fake_images = generator(torch.cat((noise, cc, dc),1))
            d_output_real = discriminator(images)
            d_output_fake = discriminator(fake_images)

            d_loss_a = -torch.mean(torch.log(d_output_real[:,0]) + torch.log(1 - d_output_fake[:,0]))

            # MI loss on latent codes
            output_cc = d_output_fake[:, 1:1+args.cc_dim]
            d_loss_cc = torch.mean((((output_cc - 0.0) / 0.5) ** 2))
            d_loss_dc = 0
            for id_dc in range(args.dc_num):
                output_dc = d_output_fake[:, 1+args.cc_dim + (id_dc * args.dc_dim):1+args.cc_dim + ((id_dc+1) * args.dc_dim)]
                dc_i = dc[:,(id_dc * args.dc_dim):((id_dc+1) * args.dc_dim)]
                d_loss_dc += -(torch.mean(torch.sum(dc_i * output_dc, 1)) + torch.mean(torch.sum(dc_i * dc_i, 1)))


            # Uncomment to include continuous code, and comment next line
            #d_loss = d_loss_a + 1.0 * d_loss_cc + 1.0 * d_loss_dc
            d_loss = d_loss_a  + 1.0 * d_loss_dc

            # gradient backpropagation and parameter update
            discriminator.zero_grad()
            d_loss.backward(retain_graph=True)
            optim_d.step()

            # training generator
            g_loss_a = -torch.mean(torch.log(d_output_fake[:,0]))

            # Uncomment to include continuous code, and comment next line
            #g_loss = g_loss_a + args.continuous_weight * d_loss_cc + 1.0 * d_loss_dc
            g_loss = g_loss_a + 1.0 * d_loss_dc

            # gradient backpropagation and parameter update
            generator.zero_grad()
            g_loss.backward()
            optim_g.step()

            # print the log info
            if (i + 1) % args.log_step == 0:
                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f'
                      % (epoch + 1, args.num_epochs, i + 1, total_num_steps, d_loss.data[0], g_loss.data[0]))

            

        # save the model parameters for each epoch
        path_g = os.path.join(args.model_path, 'generator_multi-%d.pkl' % (epoch + 1))
        path_d = os.path.join(args.model_path, 'discriminator_multi-%d.pkl' % (epoch + 1))

        torch.save(generator.state_dict(), path_g)
        torch.save(discriminator.state_dict(), d_path)

if __name__ == "__main__":
    main()
