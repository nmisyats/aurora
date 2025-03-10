import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.data import Camera
from aurora.geodesy import (
    lat_lon_to_ECEF,
    az_ze_to_UNE,
    UNE_to_ECEF,
    UNE_basis_ECEF,
    earth_radius
)

def get_device():
    return torch.device("cpu")

def create_rays(cameras: list[Camera], o_lat: float, o_lon: float):
    o_ecef = lat_lon_to_ECEF(o_lat, o_lon)

    to_o_matrix = torch.stack(UNE_basis_ECEF(o_lat, o_lon))
    to_o_matrix = torch.linalg.inv(to_o_matrix.T)
    
    ro_list, rd_list = [], []
    for cam in cameras:
        lat, lon = cam.latitude, cam.longitude

        az = torch.from_numpy(cam.azimuth).flatten()
        ze = torch.from_numpy(cam.zenith).flatten()
        rd_une = az_ze_to_UNE(az, ze)
        rd_ecef = UNE_to_ECEF(rd_une, lat, lon)
        rd_rel = torch.matmul(rd_ecef, to_o_matrix.T)

        radius = earth_radius(lat, lon) + cam.altitude
        ro_ecef = lat_lon_to_ECEF(lat, lon)
        ro_rel = radius * (ro_ecef - o_ecef)
        ro_rel = torch.matmul(ro_rel, to_o_matrix.T)
        ro_rel = ro_rel.repeat(rd_rel.shape[0], 1)

        ro_list.append(ro_rel)
        rd_list.append(rd_rel)
    
    ro = torch.cat(ro_list)
    rd = torch.cat(rd_list)
    
    return ro, rd


if __name__ == "__main__":
    from aurora.data import parse_dataset_cameras
    from pathlib import Path

    cams = parse_dataset_cameras(Path("../datasets/simulation1"))
    ro, rd = create_rays(list(cams.values()), cams["skibotn"].latitude, cams["skibotn"].longitude)
    print(ro.shape, rd.shape)