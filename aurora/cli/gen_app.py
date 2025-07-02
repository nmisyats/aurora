from pathlib import Path

import typer
from matplotlib import pyplot as plt
import torch

from aurora import models, data
from aurora.utils import choose_best_device, xy_grid, xyz_grid
import aurora.plot as aplt

# Create subcommand for generating
gen_app = typer.Typer(help="Generation utilities")

@gen_app.command("flux")
def generate_reconstructed_flux(
    recon_path: Path = typer.Argument(..., help="Path to reconstructed model"),
    res_x: int = typer.Option(128, help="X resolution"),
    res_y: int = typer.Option(128, help="Y resolution"),
    gpu: bool = typer.Option(True, help="Use GPU if available"),
    plot: bool = typer.Option(True, help="Plot the generated flux"),
    save: Path = typer.Option(None, help="Save flux data to file"),
    cmap: str = typer.Option("jet", help="Colormap for plotting"),
    ref_flux: Path = typer.Option(None, help="Reference flux to compare with"),
    ref_config: Path = typer.Option(None, help="Path to configuration file for reference flux"),
):
    """Generate flux from reconstruction using generic plotting."""
    device = choose_best_device(gpu)
    recon = models.load_model(recon_path, device)
    recon.eval()
    
    xy_min = recon.bbox.xy_min
    xy_max = recon.bbox.xy_max
    xy = xy_grid(xy_min, xy_max, res_x, res_y)
    estimated_f = recon.flux(xy).cpu()

    if save is not None:
        data.save_3d_grid_data(estimated_f, save)
    
    if plot:
        rec_xy_min = xy_min.cpu()
        rec_xy_max = xy_max.cpu()
        if ref_flux is not None and ref_config is not None:
            ref = models.load_grid_model(ref_flux, ref_config, device)
            reference_f = ref.data.cpu()
            ref_xy_min = ref.bbox.xy_min.cpu()
            ref_xy_max = ref.bbox.xy_max.cpu()
            aplt.plot_flux_2d_comparison(
                estimated_flux=estimated_f,
                reference_flux=reference_f,
                estimated_bounds=(rec_xy_min, rec_xy_max),
                reference_bounds=(ref_xy_min, ref_xy_max),
                energy_edges=recon.energy_bins.cpu(),
                cmap=cmap
            )
        else:
            aplt.plot_flux_2d(
                flux_data=estimated_f,
                xy_bounds=(recon.bbox.xy_min, recon.bbox.xy_max),
                energy_edges=recon.energy_bins.cpu(),
                cmap=cmap
            )
        plt.show()

@gen_app.command("flux-at", context_settings={"ignore_unknown_options": True})
def generate_reconstructed_flux_at(
    recon_path: Path = typer.Argument(..., help="Path to reconstructed model"),
    x: float = typer.Argument(..., help="x coordinate"),
    y: float = typer.Argument(..., help="y coordinate"),
    gpu: bool = typer.Option(True, help="Use GPU if available"),
    plot: bool = typer.Option(True, help="Plot the generated flux"),
    save: Path = typer.Option(None, help="Save flux data to file"),
    ref_flux: Path = typer.Option(None, help="Reference flux to compare with"),
    ref_config: Path = typer.Option(None, help="Path to configuration file for reference flux"),
):
    """Generate the flux curve accross energy levels at a given xy location."""
    device = choose_best_device(gpu)
    recon = models.load_model(recon_path, device)
    recon.eval()
    
    xy = torch.tensor([x, y], device=device)
    estimated_f = recon.flux(xy).cpu()

    if save is not None:
        data.save_matrix_data(estimated_f.unsqueeze(1), save)
    
    if plot:
        if ref_flux is not None and ref_config is not None:
            ref = models.load_grid_model(ref_flux, ref_config, device)
            reference_f = ref.flux(xy).cpu()
            aplt.plot_flux_1d(
                flux_data=(reference_f, estimated_f),
                energy_edges=recon.energy_bins.cpu(),
                title=f"Flux at (x, y) = ({x}, {y})",
                labels=("Reference", "Reconstructed")
            )
        else:
            aplt.plot_flux_1d(
                flux_data=estimated_f,
                energy_edges=recon.energy_bins.cpu(),
                title=f"Flux at (x, y) = ({x}, {y})"
            )
        plt.show()

@gen_app.command("emis")
def generate_volume_emission(
    recon_or_ref_path: Path = typer.Argument(..., help="Path to reconstruction or reference flux"),
    config: Path = typer.Option(None, help="Path to configuration for reference flux"),
    res_x: int = typer.Option(100, help="X resolution"),
    res_y: int = typer.Option(100, help="Y resolution"), 
    res_z: int = typer.Option(50, help="Z resolution"),
    gpu: bool = typer.Option(True, help="Use GPU if available"),
    chunk_size: int = typer.Option(16384, help="Size of chunks to split batches for flux estimation"),
    plot: bool = typer.Option(True, help="Plot the volume"),
    save: Path = typer.Option(None, help="Save volume data"),
    cmap: str = typer.Option("coolwarm", help="Volume colormap"),
    opacity: str = typer.Option("0,0.1,0.3,0.6,0.8,1.0,1.0", help="Opacity values as comma-separated list")
):
    """Generate volume emission using generic 3D plotting."""
    device = choose_best_device(gpu)
    
    if recon_or_ref_path.suffix == ".pth":
        recon = models.load_model(recon_or_ref_path, device)
        recon.eval()
        recon.chunk_size = chunk_size
        recon.chunk_progress_bar = True
        xyz_min = recon.bbox.xyz_min
        xyz_max = recon.bbox.xyz_max
        xyz = xyz_grid(xyz_min, xyz_max, res_x, res_y, res_z)
        l = recon.get_emission_rate(xyz).cpu()
    else:
        if config is None:
            typer.echo("Configuration file required for reference flux", err=True)
            raise typer.Exit(1)
        ref_recon = models.load_grid_model(recon_or_ref_path, config, device)
        xyz_min = ref_recon.bbox.xyz_min
        xyz_max = ref_recon.bbox.xyz_max
        xyz = xyz_grid(xyz_min, xyz_max, res_x, res_y, res_z)
        l = ref_recon.get_emission_rate(xyz).cpu()
    
    if save is not None:
        data.save_3d_grid_data(l, save)
    
    if plot:
        # Parse opacity values
        opacity_vals = list(map(float, opacity.split(',')))
        
        plotter = aplt.plot_volume_3d(
            volume_data=l,
            xyz_bounds=(xyz_min, xyz_max),
            scalars_name="Volume emission rate",
            cmap=cmap,
            opacity=opacity_vals
        )
        plotter.show()

@gen_app.command("dens")
def generate_electron_density():
    raise NotImplementedError

@gen_app.command("imgs")
def generate_images(
    recon_or_ref_path: Path = typer.Argument(..., help="Path to reconstruction"),
    cam_pos: Path = typer.Argument(..., help="Camera positions file"),
    cam_dir: Path = typer.Argument(..., help="Cameras directory"),
    config: Path = typer.Option(None, help="Path to configuration for reference flux"),
    locations: str = typer.Option(None, help="Comma-separated camera names"),
    ray_bins: int = typer.Option(100, help="Number of ray bins"),
    downsample: int = typer.Option(None, help="Downsample factor"),
    gpu: bool = typer.Option(True, help="Use GPU"),
    chunk_size: int = typer.Option(16384, help="Size of chunks to split batches for flux estimation"),
    plot: bool = typer.Option(True, help="Plot comparison"),
    cmap: str = typer.Option("viridis", help="Colormap"),
    figsize_per_col: float = typer.Option(2.0, help="Figure size per column")
):
    """Generate images using generic comparison plotting."""
    device = choose_best_device(gpu)
    
    if recon_or_ref_path.suffix == ".pth":
        recon = models.load_model(recon_or_ref_path, device)
        recon.chunk_size = chunk_size
        recon.chunk_progress_bar = True
    else:
        if config is None:
            typer.echo("Configuration file required for reference flux", err=True)
            raise typer.Exit(1)
        recon = models.load_grid_model(recon_or_ref_path, config, device)
    recon.eval()

    cams = data.load_cameras(cam_pos, cam_dir)
    if locations is not None:
        location_list = locations.split(',')
        cams = [c for c in cams if c.name in location_list]
    
    if downsample is not None:
        cams = [cam.downsample(downsample) for cam in cams]

    # Generate images
    generated_imgs = []
    reference_imgs = []
    camera_names = []
    
    for i, cam in enumerate(cams):
        print(f"Generating image {i+1}/{len(cams)} ({cam.name})")
        img = recon.generate_image(cam, ray_bins)
        generated_imgs.append(img)
        reference_imgs.append(cam.image)
        camera_names.append(cam.name)

    if plot:
        fig, axes_rows = aplt.plot_image_comparison(
            generated_images=generated_imgs,
            reference_images=reference_imgs,
            camera_names=camera_names,
            figsize_per_col=figsize_per_col,
            cmap=cmap
        )
        plt.show()