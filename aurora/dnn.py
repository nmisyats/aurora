import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from aurora.data import Model
from aurora.geodesy import (
    lat_lon_to_ECEF,
    az_ze_to_UNE,
    UNE_to_ECEF,
    UNE_basis_ECEF,
    earth_radius,
    inc_dec_to_UNE
)
from aurora.reconstruction import Reconstruction


class FMLP(nn.Module):
    def __init__(self, model: Model, embed_exp: int):
        assert embed_exp >= 1

        super(FMLP, self).__init__()
        self.input_size = 2
        self.embed_exp = embed_exp
        self.embed_size = self.input_size + self.input_size * 2 * embed_exp
        self.output_size = len(model.energy_bins) - 1
        
        self.fc1 = nn.Linear(self.embed_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128 + self.embed_size, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, self.output_size)
    
    def forward(self, x: torch.Tensor):
        x = self.embed_fourier(x)
        x0 = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(torch.cat([x, x0], dim=-1)))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x
    
    def embed_fourier(self, x: torch.Tensor):
        freqs = [(2**i) * torch.pi for i in range(self.embed_exp)]
        cos_x = [torch.cos(f * x) for f in freqs]
        sin_x = [torch.sin(f * x) for f in freqs]
        return torch.cat((x, *cos_x, *sin_x), dim=-1)

class LogFMLPReconstruction(Reconstruction):
    def __init__(self, pm, f_net, device):
        super().__init__(pm, f_net, device)
    
    def f(self, xy):
        xy = (xy - self.xy_min) / (self.xy_max - self.xy_min)
        log_f = self.f_net(xy)
        f = torch.pow(10.0, 7.0*log_f)
        return f 


if __name__ == "__main__":
    from aurora.data import load_dataset_description
    from aurora.plot import plot_training_loss, plot_total_energy_flux, plot_reconstructed_images, plot_volume_emission, plot_rays
    from aurora.save import save_3d_array
    from aurora.utils import downsample_camera
    from pathlib import Path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cams, pm = load_dataset_description(Path("./simulation_vertical.yaml"))

    net = FMLP(pm, 4).to(device)
    print(net)

    recon = LogFMLPReconstruction(pm, net, device)
    loss = recon.train(cams, 1000, 4096, 100)
    
    plot_training_loss(loss)

    def f_np(xy):
        xy = torch.from_numpy(xy).to(device)
        f = recon.f(xy)
        return f.detach().cpu().numpy()
    
    def L_np(p):
        p = torch.from_numpy(p).to(device)
        l = recon.L(p)
        l = l.detach().cpu().numpy()
        return l

    def img_np(cam):
        img = recon.image(downsample_camera(cam, 2), 100)
        img = img.detach().cpu().numpy()
        img = np.nan_to_num(img, nan=0.0)
        return img

    plot_total_energy_flux(f_np, pm, 128, 128)
    plot_reconstructed_images(img_np, cams)
    # L = plot_volume_emission(est_L_np, pm, 100, 100, 50)
    # save_3d_array(L, "volume_emission_rate.dat")

