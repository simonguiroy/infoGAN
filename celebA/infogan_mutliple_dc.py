import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim=128, cc_dim=1, dc_dim=10, dc_num=1):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim + cc_dim + (dc_dim * dc_num), 1024, 4, 1, 0),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.main( z )
        return out


class Discriminator(nn.Module):
    def __init__(self, cc_dim = 1, dc_dim = 10, dc_num=1):
        super(Discriminator, self).__init__()
        self.cc_dim = cc_dim
        self.dc_dim = dc_dim
        self.dc_num = dc_num

        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1 + cc_dim + (dc_dim * dc_num), 4, 1, 0)
        )

    def forward(self, x):
        out = self.main(x).squeeze()

        out[:, 0] = F.sigmoid(out[:, 0].clone())

        # the continuous code is the approximation of c itself, rather than parameters for Q(c|x)
        
        #discrete codes, each defined by a softmax
        for i in range(self.dc_num):
            #print ("i: " + str(i))
            out[:, self.cc_dim + 1 + (i * self.dc_dim):self.cc_dim + 1 + ((i+1)* self.dc_dim)] = F.softmax(out[:, self.cc_dim + 1 + (i * self.dc_dim):self.cc_dim + 1 + ((i+1) * self.dc_dim)].clone())

        return out
