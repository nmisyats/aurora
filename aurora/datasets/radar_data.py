import torch

from aurora.datasets.dataset import Dataset
from aurora.frame import Frame
import aurora.geodesy as geo


class RadarDataset(Dataset):
    SampleType = tuple[torch.Tensor, torch.Tensor]
    
    def __init__(
            self,
            altitudes: torch.Tensor,
            latitudes: torch.Tensor,
            longitudes: torch.Tensor,
            densities: torch.Tensor,
            frame: Frame,
        ):
        device = frame.device

        lats = latitudes.to(device)
        lons = longitudes.to(device)
        alts = altitudes.to(device)
        
        pts_ecef = geo.geodetic_to_ecef(lats, lons, alts)
        pts_frame = frame.from_ecef(pts_ecef, is_point=True)
        
        self.p = pts_frame
        self.d_ref = densities.to(device)

    def __len__(self):
        return len(self.p)
    
    def __getitem__(self, idx):
        return (self.p[idx], self.d_ref[idx])
    
    @property
    def device(self):
        return self.p.device
    
    def to(self, device: torch.device) -> 'RadarDataset':
        new_dataset = object.__new__(RadarDataset)
        
        new_dataset.p = self.p.to(device)
        new_dataset.d_ref = self.d_ref.to(device)

        return new_dataset