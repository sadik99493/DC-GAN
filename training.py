import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model import discriminator,generator,initialize

#configuration and hyper-parameters
latent_dim = 100
lr = 2e-4
img_size = 64
img_channels = 1
batch_size = 20
epochs = 5
d_features = 64
g_features = 64
transformations = transforms.Compose( [ transforms.Resize((img_size,img_size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize( [ 0.5 for _ in range(img_channels)] , [0.5 for _ in range(img_channels)])
                                       ] )
dataset = datasets.MNIST(root = "dataset/" , train=True , transform=transformations)

loader = DataLoader(dataset , batch_size , shuffle=True)
gen = generator(latent_dim,img_channels , g_features)
disc = discriminator(img_channels , d_features)
initialize(gen)
initialize(disc)
fixed_noise = torch.randn((32,latent_dim,1,1))  #-->for visualization

gen_optimizer = optim.Adam(gen.parameters(),lr=lr , betas = (0.5,0.999))
disc_optimizer = optim.Adam(disc.parameters(),lr=lr , betas = (0.5,0.999))
criterion = nn.BCELoss()
real_writer = SummaryWriter(f"logs/real")
fake_writer = SummaryWriter(f"logs/fake")
step=0

#training loop
gen.train()
disc.train()
for epoch in range(epochs):
    for idx,(real,_) in enumerate(loader):
        
        #---------train the discriminator---------#
        #d_loss = max( log(D_real) + log(1-D(G(noise))) )
        #BCEloss = -( y*log(y_hat) + (1-y)*log(1-y_hat) )
        #putting y=1 and y_hat = D_real
        #BCEloss = -log(D_real) --> minimizing this is same as maximising log(D_real)
        D_real = disc(real).reshape(-1)
        noise = torch.randn(batch_size,latent_dim,1,1)
        fake = gen(noise)
        D_fake = disc(fake.detach()).reshape(-1)
        loss_D_fake = criterion(D_fake,torch.zeros_like(D_fake))
        loss_D_real = criterion(D_real,torch.ones_like(D_real))
        D_loss = (loss_D_fake + loss_D_real)/2
        disc.zero_grad()
        D_loss.backward(retain_graph=True)
        disc_optimizer.step()

        #--------train the generator--------------#
        #g_loss = min( log( 1 - D(g(z))))
        out = disc(fake).reshape(-1)
        G_loss = criterion(out,torch.ones_like(out))
        gen.zero_grad()
        G_loss.backward()
        gen_optimizer.step()

        if idx%10 == 0:  #-->visualize outputs and print loss after starting of each epoch
            print(f"epoch:{epoch}/{epochs}  G_loss:{G_loss}   D_loss:{D_loss}")

            with torch.no_grad():
                fake_imgs = gen(fixed_noise)
                #real_imgs = real.reshape(-1,1,28,28)  #reshape flattened matrix
                #build grids for displaying images
                real_imgs_grid = torchvision.utils.make_grid(real[:32],normalize=True)
                fake_imgs_grid = torchvision.utils.make_grid(fake_imgs[:32],normalize=True)
                #add the images
                real_writer.add_image("real images",real_imgs_grid,global_step=step)
                fake_writer.add_image("Generated images",fake_imgs_grid,global_step=step)
            step+=1
