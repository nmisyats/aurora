from pathlib import Path

import torch

from aurora.camera import Camera
from aurora.models import FluxModel, StaticFlux
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.geometry as geom
import aurora.physics as phy
import aurora.data as data
from aurora.utils import normalize_batch_dims

class Reconstruction:
    def __init__(
            self,
            flux_model: FluxModel,
            frame: Frame,
            bbox: BBox,
            M_emis: torch.Tensor,
            M_dens: torch.Tensor,
            z_edges: torch.Tensor
        ):
        self.flux_model = flux_model
        self.M_emis = M_emis
        self.M_dens = M_dens
        self.z_edges = z_edges

        self.frame = frame
        self.bbox = bbox
        self.device = frame.device
        
        self._training = False

    @normalize_batch_dims({"xy": 1}, 0)
    def flux(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Calculate the electron flux distribution at points xy.
        Args:
            xy (torch.Tensor): Tensor of shape (n, 2) in South-East coordinates.
        
        Returns:
            torch.Tensor: Flux tensor at the points xy of shape (n, n_E)
                where n_E is the number of energy bins.
        """
        if not self._training:
            with torch.no_grad():
                return self.flux_model(xy)
        else:
            return self.flux_model(xy)

    @normalize_batch_dims({"p_frame": 1}, 0)
    def emis_rate(self, p_frame: torch.Tensor) -> torch.Tensor:
        """
        Calculate the emission rate at points p.
        
        Args:
            p_frame (torch.Tensor): Tensor of shape (n, 3) in frame coordinates.
        
        Returns:
            torch.Tensor: Emission rate tensor at the points p of shape (n,).
        """
        p_enu = self.frame.to_local_enu(p_frame, is_point=True)
        xy, z = p_frame[...,:2], p_enu[...,2]
        f = self.flux(xy)
        return phy.emis_rate(z, f, self.M_emis, self.z_edges)
    
    @normalize_batch_dims({"p_frame": 1}, 0)
    def elec_dens(self, p_frame: torch.Tensor) -> torch.Tensor:
        """
        Calculate the electron density at points p.
        
        Args:
            p_frame (torch.Tensor): Tensor of shape (n, 3) in frame coordinates.
        
        Returns:
            torch.Tensor: Emission rate tensor at the points p of shape (n,).
        """
        p_enu = self.frame.to_local_enu(p_frame, is_point=True)
        xy, z = p_frame[...,:2], p_enu[...,2]
        f = self.flux(xy)
        return phy.elec_dens(z, f, self.M_dens, self.z_edges)

    @normalize_batch_dims({"ro": 1, "rd": 1, "tn": 0, "tf": 0}, 0)
    def int_emis_ray(
            self, 
            ro: torch.Tensor, 
            rd: torch.Tensor, 
            tn: torch.Tensor, 
            tf: torch.Tensor,
            num_bins: int
        ) -> torch.Tensor:
        """
        Integrate the emission along rays along a ray.
        
        Args:
            ro (torch.Tensor): Ray origins of shape (n, 3) in frame coordinates.
            rd (torch.Tensor): Ray directions of shape (n, 3) in frame coordinates.
            tn (torch.Tensor): Near intersection distances of shape (n,).
            tf (torch.Tensor): Far intersection distances of shape (n,).
            num_bins (int): Number of bins to divide the ray in.
        
        Returns:
            torch.Tensor: Integrated emission tensor of shape (n,).
        """
        t_frame, p_frame = geom.create_ray_points(ro, rd, tn, tf, num_bins)
        p_enu = self.frame.to_local_enu(p_frame, is_point=True)
        xy, z = p_frame[...,:2], p_enu[...,2]
        f = self.flux(xy)
        l = phy.emis_rate(z, f, self.M_emis, self.z_edges)
        g = phy.int_emis_rayleigh(t_frame, l)
        # g = g * self.frame.metric_scale(rd) not required?
        return g
    
    @torch.no_grad()
    def image(self, cam: Camera, num_bins: int, nan=0.0):
        """
        Generate an image from the camera using the integrated emission along rays.
        
        Args:
            cam (Camera): Camera object to generate the image from.
            num_bins (int): Number of bins to use for ray integration.
            nan (float): Value to replace NaN values in the image.
        
        Returns:
            torch.Tensor: Image tensor of shape (cam.height, cam.width).
        """
        ro, rd = cam.create_rays_ecef(self.device)
        ro = self.frame.from_ecef(ro, is_point=True)
        rd = self.frame.from_ecef(rd, is_point=False)
        tn, tf = geom.ray_box_intersection(ro, rd, self.bbox.xyz_min, self.bbox.xyz_max)
        g = self.int_emis_ray(ro, rd, tn, tf, num_bins)
        h, w = cam.image.shape
        img = g.reshape(h, w)
        img = torch.nan_to_num(img, nan=nan)
        return img
    
    def eval(self):
        self.flux_model.eval()
        self._training = False
        return self
    
    def train(self):
        self.flux_model.train()
        self._training = True
        return self
    
    @property
    def training(self):
        return self._training


def save_reconstruction(recon: Reconstruction, path: Path | str):
    torch.save({
        "flux_model": recon.flux_model,
        "frame": recon.frame,
        "bbox": recon.bbox,
        "M_emis": recon.M_emis,
        "M_dens": recon.M_dens,
        "z_edges": recon.z_edges
    }, path)

def load_reconstruction(path: Path | str, device: torch.device):
    data = torch.load(path, map_location=device, weights_only=False)
    return Reconstruction(
        data["flux_model"],
        data["frame"],
        data["bbox"],
        data["M_emis"],
        data["M_dens"],
        data["z_edges"]
    )

def load_static_reconstruction(flux_path: Path | str, config_path: Path | str, device: torch.device):
    flux_path = Path(flux_path)
    config_path = Path(config_path)
    config = data.load_config(config_path, device)
    return Reconstruction(
        flux_model=StaticFlux(
            data=data.load_3d_grid_data(flux_path).to(device),
            E_edges=config.phys.energies,
            xy_min=config.bbox.xy_min,
            xy_max=config.bbox.xy_max,
        ).to(device),
        frame=config.frame,
        bbox=config.bbox,
        M_emis=config.phys.emis_mat,
        M_dens=config.phys.dens_mat,
        z_edges=config.phys.altitudes
    ).eval()