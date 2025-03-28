import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from matplotlib import pyplot as plt
import tqdm
from typing import Callable

from aurora.data import Camera, Model
from aurora.geodesy import (
    lat_lon_to_ECEF,
    az_ze_to_UNE,
    UNE_to_ECEF,
    UNE_basis_ECEF,
    earth_radius
)
from aurora.geometry import ray_box_intersection


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_ray_transform(pm: Model):
    o_lat, o_lon = pm.box_origin_lat, pm.box_origin_lon
    o_ecef = lat_lon_to_ECEF(o_lat, o_lon)

    to_o_une_matrix = torch.stack(UNE_basis_ECEF(o_lat, o_lon))
    to_o_une_matrix = torch.linalg.inv(to_o_une_matrix.T)
    field_dir_une = az_ze_to_UNE(
        torch.tensor([pm.mag_field_azimuth]),
        torch.tensor([90 - pm.mag_field_elevation])
    )[0]
    o_une_to_spec_matrix = torch.stack((
        torch.tensor([0.0, -1.0, 0.0]),
        torch.tensor([0.0,  0.0, 1.0]),
        field_dir_une
    ))
    o_une_to_spec_matrix = torch.linalg.inv(o_une_to_spec_matrix.T)
    to_spec_matrix = torch.matmul(o_une_to_spec_matrix, to_o_une_matrix)

    return o_ecef, to_spec_matrix

def create_camera_rays(cam: Camera, o_ecef: torch.Tensor, to_spec_matrix: torch.Tensor):
    lat, lon = cam.latitude, cam.longitude

    az = torch.from_numpy(cam.azimuth).flatten()
    ze = torch.from_numpy(cam.zenith).flatten()
    rd_une = az_ze_to_UNE(az, ze)
    rd_ecef = UNE_to_ECEF(rd_une, lat, lon)
    rd_rel = torch.matmul(rd_ecef, to_spec_matrix.T)

    radius = earth_radius(lat, lon) + cam.altitude
    ro_ecef = lat_lon_to_ECEF(lat, lon)
    ro_rel = radius * (ro_ecef - o_ecef)
    ro_rel = torch.matmul(ro_rel, to_spec_matrix.T)
    ro_rel = ro_rel.repeat(rd_rel.shape[0], 1)

    return ro_rel, rd_rel

def create_ray_g_pairs(cams: list[Camera], o_ecef: torch.Tensor, to_spec_matrix: torch.Tensor):
    ro_list, rd_list, g_ref_list = [], [], []
    for cam in cams:
        ro_rel, rd_rel = create_camera_rays(cam, o_ecef, to_spec_matrix)

        ro_list.append(ro_rel)
        rd_list.append(rd_rel)

        g = torch.from_numpy(cam.image).flatten()
        g_ref_list.append(g)
    
    ro = torch.cat(ro_list)
    rd = torch.cat(rd_list)
    g_ref = torch.cat(g_ref_list)
    
    return ro, rd, g_ref

def create_ray_points(ro: torch.Tensor, rd: torch.Tensor, min_t: float, max_t: float, num_bins: int):
    bin_edges = torch.linspace(min_t, max_t, num_bins + 1, device=device) # num_bins + 1 edges
    # Lower and upper edges of each bin
    lower_edges = bin_edges[:-1]
    upper_edges = bin_edges[1:]
    # Generate random values in each bin
    t = lower_edges + torch.rand(num_bins, device=device) * (upper_edges - lower_edges)
    # Create points
    return t, ro + t.reshape(t.shape[0], 1) * rd

def get_estimators(net: nn.Module, pm: Model, ray_bins: int):
    z_edges = torch.from_numpy(pm.altitude_bins).to(device)
    m_mat = torch.from_numpy(pm.emission_matrix).to(device)
    box_min = torch.tensor(pm.box_min).to(device)
    box_max = torch.tensor(pm.box_max).to(device)
    xy_min, xy_max = box_min[:2], box_max[:2]

    def estimate_f(xy):
        xy = (xy - xy_min) / (xy_max - xy_min)
        log_f = net(xy)
        return torch.pow(10.0, 7.0*log_f)

    def estimate_L(p):
        xy, z = p[:,:2], p[:,2]
        z_idx = torch.bucketize(z.contiguous(), z_edges) - 1
        z_idx = torch.clamp(z_idx, 0, m_mat.shape[1]-1)
        m_z = m_mat[z_idx,:]
        f = estimate_f(xy)
        l = torch.sum(m_z * f, dim=1)
        return l

    def estimate_g(ro, rd, tn, tf):
        t, p = create_ray_points(ro, rd, tn, tf, ray_bins)
        xy, z = p[:,:2], p[:,2]
        z_idx = torch.bucketize(z.contiguous(), z_edges) - 1
        z_idx = torch.clamp(z_idx, 0, m_mat.shape[1]-1)

        m_z = m_mat[z_idx,:]
        xy = (xy - xy_min) / (xy_max - xy_min)
        log_f = net(xy)
        f = torch.pow(10.0, 7.0*log_f)
        l = torch.sum(m_z * f, dim=1)
        d = t[1:] - t[:-1]
        g = torch.sum(l[1:] * d) + l[0] * (tn - t[0])
        g /= 10.0

        return g
    
    return estimate_f, estimate_L, torch.vmap(estimate_g, randomness='different')

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


def train(estimate_g: Callable, ro: torch.Tensor, rd: torch.Tensor, tn: torch.Tensor, tf: torch.Tensor, g_ref: torch.Tensor, num_iters: int, batch_size: int):
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-5, weight_decay=1.0)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

    loss_list = []

    tq = tqdm.trange(num_iters)
    for iter in tq:
        optimizer.zero_grad()

        idxs = torch.randint(0, len(ro), (batch_size,))
        g = estimate_g(ro[idxs], rd[idxs], tn[idxs], tf[idxs])
        loss = (g - g_ref[idxs])**2
        loss = loss.sum() / batch_size
        loss.backward()

        optimizer.step()
        scheduler.step()

        loss_list.append(loss.item())

        tq.set_postfix(
            loss=f"{loss.item():.0f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )
    tq.close()

    return loss_list

def plot_training_loss(loss_list: list[float]):
    plt.plot(loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training batch loss")
    plt.show()


if __name__ == "__main__":
    from aurora.data import load_dataset_description
    from aurora.plot import plot_total_energy_flux, plot_reconstructed_images, plot_volume_emission
    from aurora.save import save_3d_array
    from pathlib import Path

    cams, pm = load_dataset_description(Path("./simulation.yaml"))
    o_ecef, to_spec_matrix = get_ray_transform(pm)
    ro, rd, g_ref = create_ray_g_pairs(cams, o_ecef, to_spec_matrix)
    box_min, box_max = torch.tensor(pm.box_min), torch.tensor(pm.box_max)
    tn, tf = ray_box_intersection(ro, rd, box_min, box_max)

    # Get mask for non-NaN values in tn
    valid_mask = ~(torch.isnan(tn) | torch.isnan(tf))
    # Apply the mask to all tensors
    ro = ro[valid_mask].contiguous().to(device)
    rd = rd[valid_mask].contiguous().to(device)
    g_ref = g_ref[valid_mask].contiguous().to(device)
    tn = tn[valid_mask].contiguous().to(device)
    tf = tf[valid_mask].contiguous().to(device)

    net = FMLP(pm, 4).to(device)
    print(net)
    
    est_f, est_L, est_g = get_estimators(net, pm, 128)
    loss = train(est_g, ro, rd, tn, tf, g_ref, 10000, 4096)
    plot_training_loss(loss)

    def est_f_np(xy):
        xy = torch.from_numpy(xy).to(device)
        f = est_f(xy)
        return f.detach().cpu().numpy()
    
    def est_L_np(p):
        p = torch.from_numpy(p).to(device)
        l = est_L(p)
        l = l.detach().cpu().numpy()
        return l

    def gen_img_np(cam):
        ro, rd = create_camera_rays(cam, o_ecef, to_spec_matrix)
        tn, tf = ray_box_intersection(ro, rd, box_min, box_max)
        ro = ro.to(device)
        rd = rd.to(device)
        tn = tn.to(device)
        tf = tf.to(device)
        g = est_g(ro, rd, tn, tf)
        g = g.detach().cpu().numpy()
        h, w = cam.azimuth.shape
        g = g.reshape(h, w)
        g = np.nan_to_num(g, nan=0.0)
        return g

    plot_total_energy_flux(est_f_np, pm, 256, 256)
    plot_reconstructed_images(gen_img_np, cams)
    L = plot_volume_emission(est_L_np, pm, 100, 100, 50)
    save_3d_array(L, "volume_emission_rate.dat")

