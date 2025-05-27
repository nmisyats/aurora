import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.reconstruction import PhysicalModel, Reconstruction, RayDataset, TrainableFluxModel


class LogResMLP(TrainableFluxModel):
    def __init__(self, pm: PhysicalModel, embed_exp: int, log_scale=7.0):
        assert embed_exp >= 1

        super(LogResMLP, self).__init__()

        self.xy_min = pm.frame.xy_min
        self.xy_max = pm.frame.xy_max
        self.input_size = 2
        self.embed_exp = embed_exp
        self.embed_size = self.input_size + self.input_size * 2 * embed_exp
        self.output_size = len(pm.E_edges) - 1
        self.log_scale = log_scale
        
        self.fc1 = nn.Linear(self.embed_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128 + self.embed_size, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, self.output_size)
    
    def forward(self, xy: torch.Tensor):
        xy = (xy - self.xy_min) / (self.xy_max - self.xy_min)
        x = self.embed_fourier(xy)
        x0 = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(torch.cat([x, x0], dim=-1)))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        x = x * self.log_scale
        x = torch.pow(10.0, x)
        return x
    
    def embed_fourier(self, x: torch.Tensor):
        freqs = [(2**i) * torch.pi for i in range(self.embed_exp)]
        cos_x = [torch.cos(f * x) for f in freqs]
        sin_x = [torch.sin(f * x) for f in freqs]
        return torch.cat((x, *cos_x, *sin_x), dim=-1)


if __name__ == "__main__":
    from aurora.data import load_physical_model, load_cameras
    from aurora.plot import plot_training_loss, plot_total_energy_flux, plot_reconstructed_images, plot_volume_emission

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    pm, ref = load_physical_model("./physical_model.yaml", device)
    cams = load_cameras("./simulation.yaml")
    
    net = LogResMLP(pm, 4).to(device)
    print(net)
    recon = Reconstruction(pm, net, device)
    
    # recon = Reconstruction.load("./log_mlp_recon.pth", device)

    dataset = RayDataset(cams, pm.frame, device)
    loss = recon.train(dataset, 2000, 4096, 100)
    plot_training_loss(loss)

    # recon.save("./log_mlp_recon.pth")

    recon.eval_mode()
    plot_total_energy_flux(recon.f, pm, 128, 128, ref)
    plot_reconstructed_images(recon.image_renderer(100), cams)
    plot_volume_emission(recon.L, pm, 100, 100, 50)
