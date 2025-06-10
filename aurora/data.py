from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

from aurora.camera import Camera
from aurora.frame import Frame
import aurora.geodesy as geod
import aurora.geometry as geom



class CameraRaysDataset(Dataset):
    def __init__(self, cams: list[Camera], frame: Frame, bbox: geom.BBox):
        self.frame = frame
        self.device = frame.device

        ro_list, rd_list, g_ref_list = [], [], []
        for cam in cams:
            cam_ro, cam_rd = cam.create_rays_ecef(self.device)
            cam_ro = frame.from_ECEF(cam_ro, is_point=True)
            cam_rd = frame.from_ECEF(cam_rd, is_point=False)
            ro_list.append(cam_ro)
            rd_list.append(cam_rd)

            cam_g_ref = cam.image.flatten().to(self.device)
            g_ref_list.append(cam_g_ref)
        
        ro = torch.cat(ro_list)
        rd = torch.cat(rd_list)
        g_ref = torch.cat(g_ref_list)

        tn, tf = geom.ray_box_intersection(ro, rd, bbox.box_min, bbox.box_max)

        # Get mask for non-NaN values in tn
        valid_mask = ~(torch.isnan(tn) | torch.isnan(tf))
        # Apply the mask to all tensors
        self.ro = ro[valid_mask].contiguous()
        self.rd = rd[valid_mask].contiguous()
        self.tn = tn[valid_mask].contiguous()
        self.tf = tf[valid_mask].contiguous()
        self.g_ref = g_ref[valid_mask].contiguous()

    def __len__(self):
        return len(self.ro)
    
    def __getitem__(self, idx):
        return (
            self.ro[idx],
            self.rd[idx],
            self.tn[idx],
            self.tf[idx],
            self.g_ref[idx]
        )


class RadarPointsDataset(Dataset):
    def __init__(
            self,
            altitudes: torch.Tensor,
            latitudes: torch.Tensor,
            longitudes: torch.Tensor,
            densities: torch.Tensor,
            frame: Frame,
        ):
        self.frame = frame

        lats = latitudes.to(self.device)
        lons = longitudes.to(self.device)
        alts = altitudes.to(self.device).reshape(-1, 1)
        
        pts_ecef = geod.lat_lon_to_ECEF(lats, lons)
        pts_frame = self.frame.from_ECEF(pts_ecef, is_point=True)
        
        self.p = pts_frame
        self.d_ref = densities.to(self.device)
    
    @property
    def device(self) -> torch.device:
        return self.frame.device

    def __len__(self):
        return len(self.p)
    
    def __getitem__(self, idx):
        return (self.p[idx], self.d_ref[idx])