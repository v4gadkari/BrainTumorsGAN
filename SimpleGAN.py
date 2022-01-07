import torch 
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid, Tanh
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os 
from skimage import io
from PIL import Image








#Discriminator Class

#define layers:
#Linear Layer - img_dim
#leaky relu
#Linear Layer - img_dim, 1
#leaky_relu
#sigmoid

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)



#Generator Class

#define layers
#Linear layer - z_dim, img_dim,
#Leaky_relu
#Linear layer - img_dim, img_dim * 2
#leaky_relu
#tanh

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)





#load data
class BrainTumorsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()
        self.annotations = pd.read_csv(csv_file) #csv file of jpg path
        self.root_dir = root_dir 
        self.transform = transform 


    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index): #returns a specific sample image
        img_path  = os.path.join(self.root_dir, self.annotations['Paths'].iloc[index]) #iterate through first column, which is image file paths
        image = Image.open(img_path)


        if self.transform:
            image = self.transform(image)

        return (image) 




#set learning rate
#set epochs
#set batch size
#set discriminator optimizer
#set generator optimizer
#set bce loss function


device = "cuda" if torch.cuda.is_available() else "cpu"

img_dim = 28 * 28 * 1
z_dim = 32
learning_rate = 1e-4
epochs = 100
batch_size = 20
fixed_noise = torch.randn(batch_size, z_dim).to(device=device)

Disc = Discriminator(img_dim = img_dim).to(device=device)
Gen = Generator(z_dim = z_dim, img_dim = img_dim).to(device=device)
disc_optim = optim.Adam(Disc.parameters(), lr=learning_rate)
gen_optim = optim.Adam(Gen.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f'runs/GAN_BRAINTUMORS/fake')
writer_real = SummaryWriter(f'runs/GAN_BRAINTUMORS/real')


transformations = transforms.Compose([transforms.Resize((img_dim, img_dim)), transforms.ToTensor()])
dataset = BrainTumorsDataset(csv_file='BrainTumorPaths.csv', root_dir='glioma_tumor', transform=transformations)

data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)







#transforms

#Training Loop






step=0
for epoch in range(epochs):
    for batch_idx, sample in enumerate(data_loader):
        real = sample.view(-1, 784).to(device=device)
        batch_size = real.shape[0]


        #Train Discriminator -> maximize Loss FN:  log(D(real)) + log(1-D(G(z)))
        z = torch.randn(batch_size, z_dim).to(device=device) #generate random noise

        fake = Gen(z)#get a fake image

        disc_real = Disc(real).view(-1) #compute log(D(real))
        disc_real_loss = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = Disc(fake).view(-1) #compute log(1-D(G(z)))
        disc_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))

        total_disc_loss = (disc_real_loss + disc_fake_loss)/2

        Disc.zero_grad()
        total_disc_loss.backward(retain_graph=True)
        disc_optim.step()

    
        #Train Generator -> maximize Loss FN: log(D(G(z)))

        gen_output = Disc(fake).view(-1)
        gen_loss = criterion(gen_output, torch.ones_like(gen_output))

        Gen.zero_grad()
        gen_loss.backward()
        gen_optim.step()



        if batch_idx == 0:
            print(
                f'Epoch: {epoch}/{epochs}, \
                Loss Discriminator: {total_disc_loss:.4f},\
                Loss Generator: {gen_loss:.4f}'
            )


            with torch.no_grad():
                fake = Gen(fixed_noise).reshape(-1,1,28,28)
                data = real.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake,normalize=True)
                img_grid_real = torchvision.utils.make_grid(data,normalize=True)

                writer_fake.add_image(
                    'BrainTumor Fake Images', img_grid_fake,  global_step=step
                )

                writer_real.add_image(
                    'BrainTumor Real Images', img_grid_real, global_step=step
                )

                step += 1

            








        





"""

image_paths = []
for i in range(0, 100):
    word = f'image({i}).jpg'
    image_paths.append(word)





image_path_df = pd.DataFrame({'Paths':image_paths})

image_path_df.to_csv('BrainTumorPaths.csv')

"""