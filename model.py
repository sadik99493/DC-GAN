import torch
import torch.nn as nn

class generator(nn.Module):
    def __init__(self,latent_dim , img_channels , g_features):
        super().__init__()
        self.gen_network = nn.Sequential( self.block(latent_dim , g_features*16 , 4 , 1 , 0),   #-->half the feature maps and double the dimensions
                                         self.block(g_features*16 , g_features*8 , 4 , 2 , 1),
                                         self.block(g_features*8 , g_features*4 , 4 , 2 , 1),
                                         self.block(g_features*4 , g_features*2 , 4 , 2 , 1),
                                         nn.ConvTranspose2d(g_features*2 , img_channels , 4 , 2 , 1),
                                         nn.Tanh())

    def block(self, in_channels , out_channels , filter_size , stride , padding):
        upconv = nn.Sequential( nn.ConvTranspose2d(in_channels,out_channels,filter_size,stride,padding),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU())
        return upconv
    
    def forward(self, x):
        return self.gen_network(x)


class discriminator(nn.Module):
    def __init__(self,img_channels,d_features):
        super().__init__()
        self.disc_network = nn.Sequential( nn.Conv2d(img_channels , d_features , 4 , 2 ,1),  #-->implement first block seperately as it doesn't have batchnorm
                                  nn.LeakyReLU(0.2),
                                  self.block(d_features , d_features*2 , 2 , 1 , 4),
                                  self.block(d_features*2 , d_features*4 , 2 , 1 , 4),
                                  self.block(d_features*4 , d_features*8 , 2 , 1 , 4),
                                  nn.Conv2d(d_features*8 , 1 , 4 , 2 , 0), #-->output is a single value representing fake or real
                                  nn.Sigmoid() )

    def block(self,in_channels , out_channels , stride , padding , filter_size ):    #-->create a common block to use repeatedly
        conv = nn.Sequential( nn.Conv2d(in_channels,out_channels,filter_size,stride,padding,bias=False),
                              nn.BatchNorm2d(out_channels),
                                nn.LeakyReLU(0.2)  )
        return conv
        
    def forward(self,x):
        return self.disc_network(x)
    
def initialize(model):   #-->to initialize model weights by taking from normal dist.. of mean=0 std=0.02
    for module in model.modules():
        if isinstance(module , (nn.Conv2d , nn.ConvTranspose2d , nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data , mean = 0.0 , std = 0.02)

#-->testing the model outputs || ignore this part
def test():
    x = torch.randn((8,3,64,64))
    disc = discriminator(3,8)
    initialize(disc)
    print(disc(x).shape)
    gen = generator(100 , 3 , 8)
    initialize(gen)
    y = torch.randn((8,100,1,1))
    print(gen(y).shape)

if __name__ == "__main__":
    test()