from pathlib import Path

import typer
from matplotlib import pyplot as plt


# Create subcommand for plotting
plot_app = typer.Typer(help="Plotting utilities")

@plot_app.command("flux")
def plot_flux(
    flux_data: Path = typer.Argument(..., help="Path to flux data"),
    config_path: Path = typer.Argument(..., help="Path to configuration YAML file"),
    title: str = typer.Option("$Q_0$", help="Plot title"),
    cmap: str = typer.Option("jet", help="Colormap name"),
    figsize: str = typer.Option("8,6", help="Figure size as 'width,height'")
):
    """Plot the total energy flux of saved flux data."""
    import aurora as au
    import aurora.physics as phy

    config = au.data.load_config(config_path)
    f = au.data.load_3d_grid_data(flux_data)
    q0 = phy.compute_total_energy_flux(f, config.energy_bins)
    
    width, height = map(float, figsize.split(','))
    
    au.plot.plot_flux_2d(
        flux_data=q0,
        xy_bounds=config.bbox.xy_bounds,
        title=title,
        cmap=cmap,
        figsize=(width, height)
    )
    plt.show()

@plot_app.command("flux-at", context_settings={"ignore_unknown_options": True})
def plot_flux_at(
    flux_data: Path = typer.Argument(..., help="Path to flux data"),
    config_path: Path = typer.Argument(..., help="Path to configuration YAML file"),
    x: float = typer.Argument(..., help="x coordinate"),
    y: float = typer.Argument(..., help="y coordinate"),
    plot: bool = typer.Option(True, help="Plot the generated flux"),
    save: Path = typer.Option(None, help="Save flux data to file"),
):
    """Plots the flux curve accross energy levels at a given xy location from saved flux data."""
    import torch
    import aurora as au

    recon = au.models.load_grid_model(flux_data, config_path)
    f = recon.flux(torch.tensor([x, y]))

    if save is not None:
        au.data.save_matrix_data(f.unsqueeze(1), save)
    
    if plot:
        au.plot.plot_flux_1d(
            flux_data=f,
            energy_edges=recon.energy_bins,
            title=f"Flux at (x, y) = ({x}, {y})"
        )
        plt.show()

@plot_app.command("emis")
def plot_volume_emission(
    emis_data_path: Path = typer.Argument(..., help="Path to emission rate data"),
    config_path: Path = typer.Argument(..., help="Path to configuration YAML file"),
    save: Path = typer.Option(None, help="Save plot to file"),
):
    """Plot saved 3D volume emission rate."""
    import aurora as au

    emis_data = au.data.load_3d_grid_data(emis_data_path)
    config = au.data.load_config(config_path)
    pl = au.plot.plot_volume_3d(
        volume_data=10**6 * emis_data,
        xyz_bounds=(config.bbox.xyz_min, config.bbox.xyz_max),
        scalars_name="L [photons/m³/s]"
    )

    if save is not None:
        pl.save_graphic(
            filename=save,
            title="3D volume emission rate"
        )
    
    pl.show()

@plot_app.command("dens")
def plot_electron_density(
    dens_data_path: Path = typer.Argument(..., help="Path to emission rate data"),
    config_path: Path = typer.Argument(..., help="Path to configuration YAML file"),
    save: Path = typer.Option(None, help="Save plot to file"),
):
    """Plot saved 3D electron density."""
    import aurora as au

    dens_data = au.data.load_3d_grid_data(dens_data_path)
    config = au.data.load_config(config_path)
    pl = au.plot.plot_volume_3d(
        volume_data=10**6 * dens_data,
        xyz_bounds=(config.bbox.xyz_min, config.bbox.xyz_max),
        scalars_name="D [electrons/m³]"
    )

    if save is not None:
        pl.save_graphic(
            filename=save,
            title="3D electron density"
        )

    pl.show()

@plot_app.command("cams")
def plot_cameras(
    cams: Path = typer.Argument(..., help="Camera dataset directory"),
    location: str = typer.Option(None, help="Name of specific camera to plot"),
    title_format: str = typer.Option("{name} ({latitude:.3f}°N {longitude:.3f}°E +{altitude:.3f}km)", help="Title format"),
    cmap: str = typer.Option("viridis", help="Colormap name"),
    figsize_per_image: float = typer.Option(3.0, help="Size factor per image"),
    max_cols: int = typer.Option(None, help="Maximum columns in grid")
):
    """Plot camera images from a dataset."""
    import aurora as au

    cam_pos = cams / "camera_position.set"
    cam_dir = cams
    cams = au.data.load_cameras(cam_pos, cam_dir)

    au.plot.plot_cameras_grid(
        cameras=cams,
        selected_camera=location,
        figsize_per_image=figsize_per_image,
        cmap=cmap,
        title_format=title_format,
        max_cols=max_cols
    )
    plt.show()

@plot_app.command("radar")
def plot_radar(
    radar_data: Path = typer.Argument(..., help="Radar data file"),
    config_path: Path = typer.Argument(..., help="Path to configuration YAML file"),
):
    """Plot camera images from a dataset."""
    import aurora as au

    radar = au.data.load_radar_point_cloud(radar_data)
    config = au.data.load_config(config_path)
    pts_ecef = au.geodesy.geodetic_to_ecef(radar.latitudes, radar.longitudes, radar.altitudes)
    pts_frame = config.frame.from_ecef(pts_ecef, is_point=True)
    dens = radar.densities
    au.plot.plot_3d_scatter(pts_frame, dens / 10**5, unit=r"$10^5 \mathrm{cm}^{-3}$",
                        #  xlim=(config.bbox.x_min, config.bbox.x_max),
                        #  ylim=(config.bbox.y_min, config.bbox.y_max),
                        #  zlim=(config.bbox.z_min, config.bbox.z_max)
                        )
    plt.show()