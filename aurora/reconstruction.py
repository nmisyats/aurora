# from pathlib import Path
# from abc import ABC, abstractmethod

# import torch
# import torch.nn as nn
# import tqdm

# from aurora.camera import Camera
# from aurora.models import FluxModel, GridSampledFlux
# from aurora.frame import Frame
# from aurora.bbox import BBox
# import aurora.geometry as geom
# import aurora.physics as phy
# import aurora.data as data
# from aurora.utils import normalize_batch_dims, ceiled_div, iter_chunks


# def save_reconstruction(recon: Reconstruction, path: Path | str):
#     torch.save({
#         "flux_model": recon.flux_model,
#         "frame": recon.frame,
#         "bbox": recon.bbox,
#         "emis_mat": recon.emis_mat,
#         "dens_mat": recon.dens_mat,
#         "altitude_bins": recon.altitude_bins
#     }, path)

# def load_reconstruction(path: Path | str, device: torch.device):
#     data = torch.load(path, map_location=device, weights_only=False)
#     return Reconstruction(
#         data["flux_model"],
#         data["frame"],
#         data["bbox"],
#         data["emis_mat"],
#         data["dens_mat"],
#         data["altitude_bins"]
#     )

# def load_static_reconstruction(flux_path: Path | str, config_path: Path | str, device: torch.device):
#     flux_path = Path(flux_path)
#     config_path = Path(config_path)
#     config = data.load_config(config_path, device)
#     return Reconstruction(
#         flux_model=GridSampledFlux(
#             xy_min=config.bbox.xy_min,
#             xy_max=config.bbox.xy_max,
#             energy_bins=config.physics.energy_bins,
#             data=data.load_3d_grid_data(flux_path).to(device)
#         ).to(device),
#         frame=config.frame,
#         bbox=config.bbox,
#         emis_mat=config.physics.emis_mat,
#         dens_mat=config.physics.dens_mat,
#         altitude_bins=config.physics.altitude_bins
#     ).eval()