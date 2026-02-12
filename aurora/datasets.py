from abc import ABC, abstractmethod
from typing import NamedTuple, List, Optional
from dataclasses import dataclass, astuple

import torch

from aurora.camera import Camera
from aurora.bbox import BBox
import aurora.geodesy as geo
from aurora.data import RadarPointCloud


@dataclass
class DatasetBatch(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of samples in the batch.
        """
        ...
    
    def __iter__(self):
        """
        Iterates through the members as a tuple.
        """
        return iter(astuple(self))
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...
    
    @abstractmethod
    def to(self, *args, **kwargs) -> 'DatasetBatch':
        ...


class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, idx) -> DatasetBatch:
        """
        Get batch(es).
        
        Args:
            idx: int, slice, list, or tensor of indices
            
        Returns:
            Batch of the selected indices.
        """
        ...
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
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
            idx = torch.randint(0, len(self), (batch_size,))
        else:
            perm = torch.randperm(len(self))
            idx = perm[:batch_size]
        return self[idx]
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...
    
    @abstractmethod
    def to(self, *args, **kwargs) -> 'Dataset':
        ...

@dataclass
class RayBatch(DatasetBatch):
    ro: torch.Tensor
    rd: torch.Tensor
    tn: torch.Tensor
    tf: torch.Tensor
    g_ref: torch.Tensor
    wl: str

    def __len__(self):
        return len(self.ro)
    
    @property
    def device(self):
        return self.ro.device
    
    def to(self, *args, **kwargs):
        self.ro = self.ro.to(*args, **kwargs)
        self.rd = self.rd.to(*args, **kwargs)
        self.tn = self.tn.to(*args, **kwargs)
        self.tf = self.tf.to(*args, **kwargs)
        self.g_ref = self.g_ref.to(*args, **kwargs)
        return self

class RayDataset(Dataset):
    def __init__(self, cameras: List[Camera], bbox: BBox, wl: Optional[str] = None):
        if len(cameras) == 0:
            raise ValueError("Empty camera list.")
        
        device = bbox.device

        ro_list, rd_list, g_ref_list = [], [], []
        for cam in filter(lambda c: c.wavelength == wl, cameras):
            cam_ro, cam_rd = cam.create_rays_ecef(device)
            cam_ro = bbox.frame.from_ecef(cam_ro, is_point=True)
            cam_rd = bbox.frame.from_ecef(cam_rd, is_point=False)
            ro_list.append(cam_ro)
            rd_list.append(cam_rd)

            cam_g_ref = cam.image.flatten().to(device)
            g_ref_list.append(cam_g_ref)
        
        if len(ro_list) == 0:
            raise ValueError(f"Empty dataset for wavelength {wl}.")

        ro = torch.cat(ro_list)
        rd = torch.cat(rd_list)
        g_ref = torch.cat(g_ref_list)

        tn, tf = bbox.intersection(ro, rd)
        
        valid_mask = ~(torch.isnan(tn) | torch.isnan(tf))

        # Apply the mask to all tensors
        self.ro = ro[valid_mask].contiguous()
        self.rd = rd[valid_mask].contiguous()
        self.tn = tn[valid_mask].contiguous()
        self.tf = tf[valid_mask].contiguous()
        self.g_ref = g_ref[valid_mask].contiguous()
        self.wl = wl
    
    def __getitem__(self, idx):
        return RayBatch(
            self.ro[idx],
            self.rd[idx],
            self.tn[idx],
            self.tf[idx],
            self.g_ref[idx],
            self.wl
        )
    
    def __len__(self):
        return len(self.ro)
    
    @property
    def device(self):
        return self.ro.device
    
    def to(self, *args, **kwargs):
        self.ro = self.ro.to(*args, **kwargs)
        self.rd = self.rd.to(*args, **kwargs)
        self.tn = self.tn.to(*args, **kwargs)
        self.tf = self.tf.to(*args, **kwargs)
        self.g_ref = self.g_ref.to(*args, **kwargs)
        return self

@dataclass
class RadarBatch(DatasetBatch):
    p: torch.Tensor
    d_ref: torch.Tensor

    def __len__(self):
        return len(self.p)
    
    @property
    def device(self):
        return self.p.device
    
    def to(self, *args, **kwargs):
        self.p = self.p.to(*args, **kwargs)
        self.d_ref = self.d_ref.to(*args, **kwargs)
        return self

class RadarDataset(Dataset):
    def __init__(self, data: RadarPointCloud, bbox: BBox):
        if len(data.latitudes) == 0:
            raise ValueError("Empty radar cloud.")

        device = bbox.device

        lat = data.latitudes.to(device)
        lon = data.longitudes.to(device)
        h = data.altitudes.to(device)
        d = data.densities.to(device)
        
        p_ecef = geo.geodetic_to_ecef(lat, lon, h)
        p = bbox.frame.from_ecef(p_ecef, is_point=True)

        inside_mask = bbox.contains(p)

        self.p = p[inside_mask].contiguous()
        self.d_ref = d[inside_mask].contiguous()
    
    def __getitem__(self, idx):
        return RadarBatch(self.p[idx], self.d_ref[idx])
    
    def __len__(self):
        return len(self.p)
    
    @property
    def device(self):
        return self.p.device
    
    def to(self, *args, **kwargs):
        self.p = self.p.to(*args, **kwargs)
        self.d_ref = self.d_ref.to(*args, **kwargs)
        return self