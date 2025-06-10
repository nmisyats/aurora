from abc import ABC, abstractmethod

import torch

from aurora.camera import Camera
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.geodesy as geo
import aurora.geometry as geom


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        ...

    @abstractmethod
    def __getitem__(self, idx) -> tuple[torch.Tensor, ...]:
        """
        Get sample(s) from the dataset at index idx.
        
        Parameters:
            idx: Index of the sample to retrieve.
        
        Returns:
            A tuple containing the sample data.
        """
        ...
    
    def random_sample(self, num_samples: int):
        """
        Sample a number of random samples from the dataset.
        
        Parameters:
            num_samples: Number of samples to retrieve.
        
        Returns:
            A list of tuples containing the sampled data.
        """
        if num_samples > len(self):
            raise ValueError("num_samples exceeds dataset size")
        indices = torch.randint(0, len(self), (num_samples,))
        return self[indices]


class CameraRaysDataset(Dataset):
    def __init__(self, cams: list[Camera], frame: Frame, bbox: BBox):
        self.frame = frame
        self.device = frame.device

        ro_list, rd_list, g_ref_list = [], [], []
        for cam in cams:
            cam_ro, cam_rd = cam.create_rays_ecef(self.device)
            cam_ro = frame.from_ecef(cam_ro, is_point=True)
            cam_rd = frame.from_ecef(cam_rd, is_point=False)
            ro_list.append(cam_ro)
            rd_list.append(cam_rd)

            cam_g_ref = cam.image.flatten().to(self.device)
            g_ref_list.append(cam_g_ref)
        
        ro = torch.cat(ro_list)
        rd = torch.cat(rd_list)
        g_ref = torch.cat(g_ref_list)

        tn, tf = geom.ray_box_intersection(ro, rd, bbox.xyz_min, bbox.xyz_max)

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
        self.device = frame.device

        lats = latitudes.to(self.device)
        lons = longitudes.to(self.device)
        alts = altitudes.to(self.device)
        
        pts_ecef = geo.geodetic_to_ecef(lats, lons, alts)
        pts_frame = self.frame.from_ecef(pts_ecef, is_point=True)
        
        self.p = pts_frame
        self.d_ref = densities.to(self.device)

    def __len__(self):
        return len(self.p)
    
    def __getitem__(self, idx):
        return (self.p[idx], self.d_ref[idx])