import torch

from aurora.datasets.dataset import Dataset
from aurora.camera import Camera
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.geometry as geom


class RayDataset(Dataset):
    SampleType = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    def __init__(self, cams: list[Camera], frame: Frame, bbox: BBox):
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
