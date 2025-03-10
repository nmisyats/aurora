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

def change_vector_basis(v, bx, by, bz):
    vx, vy, vz = v
    new_vx = vx * bx[0] + vy * bx[1] + vz * bx[2]
    new_vy = vx * by[0] + vy * by[1] + vz * by[2]
    new_vz = vx * bz[0] + vy * bz[1] + vz * bz[2]
    return new_vx, new_vy, new_vz

def create_rays(cameras: list[Camera], o_lat: float, o_lon: float):
    T = lambda x: torch.tensor([x], device=get_device())

    o_ecef = lat_lon_to_ECEF(T(o_lat), T(o_lon))
    o_une_ecef_basis = UNE_basis_ECEF(T(o_lat), T(o_lon))
    
    ro_list, rd_list = [], []
    for cam in cameras:
        lat, lon = T(cam.latitude), T(cam.longitude)

        az = torch.from_numpy(cam.azimuth).flatten().to(get_device())
        ze = torch.from_numpy(cam.zenith).flatten().to(get_device())
        rd_une = az_ze_to_UNE(az, ze)
        rd_ecef = UNE_to_ECEF(rd_une, lat, lon)
        rd_rel = change_vector_basis(rd_ecef, *o_une_ecef_basis)

        radius = earth_radius(lat.item(), lon.item()) + cam.altitude
        ro_ecef = lat_lon_to_ECEF(lat, lon)
        ro_rel = change_vector_basis((
                radius * (ro_ecef[0] - o_ecef[0]),
                radius * (ro_ecef[1] - o_ecef[1]),
                radius * (ro_ecef[2] - o_ecef[2]),
            ), *o_une_ecef_basis)

        ro_list.append(torch.stack(ro_rel).repeat((1, rd_rel[0].shape[0])))
        rd_list.append(torch.stack(rd_rel))
    
    ro = torch.cat(ro_list)
    rd = torch.cat(rd_list)
    
    return ro, rd


if __name__ == "__main__":
    from aurora.data import parse_dataset_cameras
    from pathlib import Path

    cams = parse_dataset_cameras(Path("../datasets/simulation1"))
    ro, rd = create_rays(list(cams.values()), 0, 0)
    print(ro.shape, rd.shape)