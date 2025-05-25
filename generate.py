# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
import torch
import numpy as np
from config import N_FEAT, N_CFEAT, HEIGHT, TIMESTEPS, BETA1, BETA2, SAVE_DIR
from model import ContextUnet
from torchvision.utils import save_image
from utils import save_intermediate_grids
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))

# construct DDPM noise schedule
b_t = (BETA2 - BETA1) * torch.linspace(0, 1, TIMESTEPS + 1, device=device) + BETA1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
ab_t[0] = 1

# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise


@torch.no_grad()
def sample_ddpm(n_sample, nn_model, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, HEIGHT, HEIGHT).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    print(f'timesteps: {TIMESTEPS}')
    for i in range(TIMESTEPS, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / TIMESTEPS])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t)    # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate ==0 or i==TIMESTEPS or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate


if __name__ == "__main__":
    
    nn_model = ContextUnet(3, N_FEAT, N_CFEAT, HEIGHT).to(device)
    # load in model weights and set to eval mode
    nn_model.load_state_dict(torch.load(f"{SAVE_DIR}/model_trained.pth", map_location=device))
    nn_model.eval()
    print("Loaded in Model")   
    samples, intermediate = sample_ddpm(n_sample=20, nn_model=nn_model)
    
    # Create directory for intermediate images if it doesn't exist
    os.makedirs("intermediate_images", exist_ok=True)    
    save_intermediate_grids(intermediate)