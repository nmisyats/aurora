import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from aurora.model import PhysicalModel
from aurora.reconstruction import Reconstruction
from aurora.utils import numpify


class LogResMLP(nn.Module):
    def __init__(self, output_size: int, embed_exp: int):
        assert embed_exp >= 1

        super(LogResMLP, self).__init__()
        self.input_size = 2
        self.embed_exp = embed_exp
        self.embed_size = self.input_size + self.input_size * 2 * embed_exp
        self.output_size = output_size
        
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


class LogMLPReconstruction(Reconstruction):
    def __init__(self, pm: PhysicalModel, f_net: nn.Module, device: torch.device):
        super().__init__(pm, device)
        self.f_net = f_net
    
    def parameters(self):
        return self.f_net.parameters()
    
    def f(self, xy):
        xy = (xy - self.xy_min) / (self.xy_max - self.xy_min)
        log_f = self.f_net(xy)
        f = torch.pow(10.0, 7.0*log_f)
        return f


if __name__ == "__main__":
    from aurora.data import load_dataset_description
    from aurora.plot import plot_training_loss, plot_total_energy_flux, plot_reconstructed_images, plot_volume_emission, plot_rays
    from aurora.save import save_3d_array
    from pathlib import Path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cams, pm = load_dataset_description(Path("./simulation.yaml"))

    net = LogResMLP(len(pm.energy_bins) - 1, 4).to(device)
    print(net)

    recon = LogMLPReconstruction(pm, net, device)
    loss = recon.train(cams, 2000, 4096, 100)
    
    plot_training_loss(loss)

    @numpify(device=device)
    def f(xy):
        return recon.f(xy)
    
    @numpify(device=device)
    def L(p):
        return recon.L(p)

    @numpify(device=device)
    def image(cam):
        return recon.image(cam, 100)

    plot_total_energy_flux(f, pm, 128, 128)
    plot_reconstructed_images(image, cams)
    # L = plot_volume_emission(est_L_np, pm, 100, 100, 50)
    # save_3d_array(L, "volume_emission_rate.dat")

