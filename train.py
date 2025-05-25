from generate import perturb_input
from model import ContextUnet
from torch.utils.data import DataLoader
from dataset import CustomDataset
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from config import *
from dataset import transform

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nn_model = ContextUnet(in_channels=3, n_feat=N_FEAT, n_cfeat=N_CFEAT, height=HEIGHT).to(device)

# load dataset and construct optimizer
dataset = CustomDataset("../datasets/sprites/sprites_1788_16x16.npy", "../datasets/sprites/sprite_labels_nc_1788_16x16.npy", transform, null_context=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
optim = torch.optim.Adam(nn_model.parameters(), lr=LRATE)

# training without context code

# set into train mode
nn_model.train()

for ep in range(N_EPOCH):
    print(f'epoch {ep}')
    
    # linearly decay learning rate
    optim.param_groups[0]['lr'] = LRATE*(1-ep/N_EPOCH)
    
    pbar = tqdm(dataloader, mininterval=2 )
    for x, _ in pbar:   # x: images
        optim.zero_grad()
        x = x.to(device)
        
        # perturb data
        noise = torch.randn_like(x)
        t = torch.randint(1, TIMESTEPS + 1, (x.shape[0],)).to(device) 
        x_pert = perturb_input(x, t, noise)
        
        # use network to recover noise
        pred_noise = nn_model(x_pert, t / TIMESTEPS)
        
        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        
        optim.step()

    # save model periodically
    if ep%4==0 or ep == int(N_EPOCH-1):
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        torch.save(nn_model.state_dict(), SAVE_DIR + f"model_{ep}.pth")
        print('saved model at ' + SAVE_DIR + f"model_{ep}.pth")