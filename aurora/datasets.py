from abc import ABC, abstractmethod
from typing import NamedTuple, List

import torch

from aurora.camera import Camera
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.geodesy as geo
import aurora.geometry as gmt
from aurora.data import RadarData


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        ...

    @abstractmethod
    def __getitem__(self, idx) -> NamedTuple:
        """
        Get sample(s) from the dataset at index idx.
        
        Args:
            idx: Index of the sample to retrieve.
        
        Returns:
            A tuple containing the sample data.
        """
        ...
    
    def sample_batch(self, batch_size: int, replacement=True):
        """
        Sample a number of random samples from the dataset.
        
        Args:
            batch_size: Number of samples to retrieve.
            replacement: Whether to sample with replacement.
        
        Returns:
            Tuples of tensors containing the sampled data.
        """
        assert batch_size > 0 and batch_size <= len(self)
        if replacement:
            indices = torch.randint(0, len(self), (batch_size,))
        else:
            perm = torch.randperm(len(self))
            indices = perm[:batch_size]
        return self[indices]
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...
    
    @abstractmethod
    def to(self, device: torch.device) -> 'Dataset':
        ...


class RayBatch(NamedTuple):
    ro: torch.Tensor
    rd: torch.Tensor
    tn: torch.Tensor
    tf: torch.Tensor
    g_ref: torch.Tensor

class RayDataset(Dataset):
    def __init__(self, cams: List[Camera], frame: Frame, bbox: BBox):
        device = frame.device

        ro_list, rd_list, g_ref_list = [], [], []
        for cam in cams:
            cam_ro, cam_rd = cam.create_rays_ecef(device)
            cam_ro = frame.from_ecef(cam_ro, is_point=True)
            cam_rd = frame.from_ecef(cam_rd, is_point=False)
            ro_list.append(cam_ro)
            rd_list.append(cam_rd)

            cam_g_ref = cam.image.flatten().to(device)
            g_ref_list.append(cam_g_ref)
        
        ro = torch.cat(ro_list)
        rd = torch.cat(rd_list)
        g_ref = torch.cat(g_ref_list)

        tn_bbox, tf_bbox = bbox.intersection(ro, rd)
        bbox_mask = ~(torch.isnan(tn_bbox) | torch.isnan(tf_bbox))

        slice_min = torch.tensor([-torch.inf, -torch.inf, bbox.z_min], device=device)
        slice_max = torch.tensor([ torch.inf,  torch.inf, bbox.z_max], device=device)
        tn, tf = gmt.ray_box_intersection(ro, rd, slice_min, slice_max)

        # Apply the mask to all tensors
        self.ro = ro[bbox_mask].contiguous()
        self.rd = rd[bbox_mask].contiguous()
        self.tn = tn[bbox_mask].contiguous()
        self.tf = tf[bbox_mask].contiguous()
        self.g_ref = g_ref[bbox_mask].contiguous()

    def __len__(self):
        return len(self.ro)
    
    def __getitem__(self, idx):
        return RayBatch(
            self.ro[idx],
            self.rd[idx],
            self.tn[idx],
            self.tf[idx],
            self.g_ref[idx]
        )
    
    @property
    def device(self):
        return self.ro.device
    
    def to(self, device: torch.device) -> 'RayDataset':
        new_dataset = object.__new__(RayDataset)
        
        new_dataset.ro = self.ro.to(device)
        new_dataset.rd = self.rd.to(device)
        new_dataset.tn = self.tn.to(device)
        new_dataset.tf = self.tf.to(device)
        new_dataset.g_ref = self.g_ref.to(device)

        return new_dataset


class RadarBatch(NamedTuple):
    p: torch.Tensor
    d_ref: torch.Tensor

class RadarDataset(Dataset):
    def __init__(
            self,
            data: RadarData,
            frame: Frame,
        ):
        device = frame.device

        lats = data.latitudes.to(device)
        lons = data.longitudes.to(device)
        alts = data.altitudes.to(device)
        
        pts_ecef = geo.geodetic_to_ecef(lats, lons, alts)
        pts_frame = frame.from_ecef(pts_ecef, is_point=True)
        
        self.p = pts_frame
        self.d_ref = data.densities.to(device)

    def __len__(self):
        return len(self.p)
    
    def __getitem__(self, idx):
        return RadarBatch(self.p[idx], self.d_ref[idx])
    
    @property
    def device(self):
        return self.p.device
    
    def to(self, device: torch.device) -> 'RadarDataset':
        new_dataset = object.__new__(RadarDataset)
        
        new_dataset.p = self.p.to(device)
        new_dataset.d_ref = self.d_ref.to(device)

        return new_dataset