import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.data import Camera, Model
from aurora.geodesy import (
    lat_lon_to_ECEF,
    az_ze_to_UNE,
    UNE_to_ECEF,
    UNE_basis_ECEF,
    earth_radius
)
from aurora.geometry import ray_box_intersection

def get_device():
    return torch.device("cpu")

def create_rays(cameras: list[Camera], model: Model, o_lat: float, o_lon: float):
    o_ecef = lat_lon_to_ECEF(o_lat, o_lon)

    to_o_une_matrix = torch.stack(UNE_basis_ECEF(o_lat, o_lon))
    to_o_une_matrix = torch.linalg.inv(to_o_une_matrix.T)
    field_dir_une = az_ze_to_UNE(
        torch.tensor([model.field_azimuth]),
        torch.tensor([90 - model.field_elevation])
    )[0]
    o_une_to_spec_matrix = torch.stack((
        torch.tensor([0.0, -1.0, 0.0]),
        torch.tensor([0.0,  0.0, 1.0]),
        field_dir_une
    ))
    o_une_to_spec_matrix = torch.linalg.inv(o_une_to_spec_matrix.T)
    to_spec_matrix = torch.matmul(o_une_to_spec_matrix, to_o_une_matrix)
    
    ro_list, rd_list = [], []
    for cam in cameras:
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

        ro_list.append(ro_rel)
        rd_list.append(rd_rel)
    
    ro = torch.cat(ro_list)
    rd = torch.cat(rd_list)
    
    return ro, rd

def create_ray_points(ro: torch.Tensor, rd: torch.Tensor, min_t: float, max_t: float):
    num_bins = 128
    bin_edges = torch.linspace(min_t, max_t, num_bins + 1) # num_bins + 1 edges
    # Lower and upper edges of each bin
    lower_edges = bin_edges[:-1]
    upper_edges = bin_edges[1:]
    # Generate random values in each bin
    t = lower_edges + torch.rand(num_bins) * (upper_edges - lower_edges)
    # Create points
    return t, ro + t.reshape(t.shape[0], 1) * rd

if __name__ == "__main__":
    from aurora.data import parse_dataset_cameras, parse_model_data
    from pathlib import Path

    cams = parse_dataset_cameras(Path("../datasets/simulation1"))
    model = parse_model_data(Path("../model"))
    ro, rd = create_rays(list(cams.values()), model, cams["skibotn"].latitude, cams["skibotn"].longitude)
    print(ro.shape, rd.shape)
    box_min, box_max = torch.tensor(model.box_min), torch.tensor(model.box_max)
    print(box_min.shape, box_max.shape)
    tn, tf = ray_box_intersection(ro, rd, box_min, box_max)
    print(tn.shape, tf.shape)
    
    i = 0
    t, p = create_ray_points(ro[i], rd[i], tn[i], tf[i])
    print(t.shape, p.shape)