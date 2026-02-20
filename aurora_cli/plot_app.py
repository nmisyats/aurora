from pathlib import Path
from typing import Literal, Optional

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
    camera: Optional[str] = typer.Option(
        None,
        "--camera",
        "-c",
        help="Camera selector: <location>[:wavelength], <location>, or folder name",
    ),
    location: Optional[str] = typer.Option(
        None,
        "--location",
        help="Alias for --camera",
    ),
    wavelength: Optional[str] = typer.Option(
        None,
        "--wavelength",
        "-w",
        help="Load only cameras with this wavelength",
    ),
    image_index: int = typer.Option(
        0,
        "-i",
        min=0,
        help="Image index to plot from each camera",
    ),
    anim: Optional[str] = typer.Option(
        None,
        "--anim",
        help="Animate an inclusive image range 'start,end' (example: 5,15)",
    ),
    anim_interval_ms: int = typer.Option(
        200,
        "--anim-interval-ms",
        min=1,
        help="Animation frame interval in milliseconds",
    ),
    anim_repeat: bool = typer.Option(
        True,
        "--anim-repeat/--no-anim-repeat",
        help="Repeat animation when it reaches the end",
    ),
    title_format: str = typer.Option(
        "{camera_id}",
        help="Title format, e.g. '{camera_id}' or '{location} {date_time}'",
    ),
    cmap: str = typer.Option("viridis", help="Colormap name"),
    figsize_per_image: float = typer.Option(3.0, help="Size factor per image"),
    max_cols: Optional[int] = typer.Option(None, help="Maximum columns in grid"),
    global_color_scale: bool = typer.Option(
        True,
        "--global-color-scale/--per-camera-color-scale",
        help="Use shared color scale across all plotted images",
    ),
):
    """Plot camera images from a dataset."""
    import aurora as au

    if camera is not None and location is not None and camera != location:
        raise typer.BadParameter("Use either --camera or --location, not both with different values.")
    selected_camera = camera if camera is not None else location
    wl_filter = [wavelength] if wavelength is not None else None
    anim_range = None
    if anim is not None:
        parts = [part.strip() for part in anim.split(",")]
        if len(parts) != 2:
            raise typer.BadParameter("--anim must be formatted as 'start,end' (example: 5,15).")
        try:
            start_idx = int(parts[0])
            end_idx = int(parts[1])
        except ValueError as exc:
            raise typer.BadParameter("--anim values must be integers (example: 5,15).") from exc
        if start_idx < 0 or end_idx < 0:
            raise typer.BadParameter("--anim indices must be >= 0.")
        if end_idx < start_idx:
            raise typer.BadParameter("--anim end index must be >= start index.")
        anim_range = (start_idx, end_idx)

    cams = au.data.load_cameras(cams, wl_filter=wl_filter)

    au.plot.plot_cameras_grid(
        cameras=cams,
        selected_camera=selected_camera,
        image_index=image_index,
        anim_range=anim_range,
        anim_interval_ms=anim_interval_ms,
        anim_repeat=anim_repeat,
        figsize_per_image=figsize_per_image,
        cmap=cmap,
        title_format=title_format,
        global_color_scale=global_color_scale,
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

@plot_app.command("mat")
def plot_model_matrices(
    model_dir: Path = typer.Argument(..., help="Path to model directory containing matrix .dat files"),
    matrix_type: Literal["emis", "dens"] = typer.Option(..., "--type", help="Matrix type to load"),
    cmap: str = typer.Option("viridis", help="Colormap name"),
    figsize: str = typer.Option("10,5", help="Figure size as 'width,height'"),
):
    """Plot emission or density matrices found in a model folder."""
    import aurora as au

    model_dir = Path(model_dir)
    altitude_path = model_dir / "altitude.dat"
    energy_path = model_dir / "energy.dat"

    if not altitude_path.exists() or not energy_path.exists():
        raise ValueError("Missing altitude.dat or energy.dat in model directory.")

    matrix_files = sorted(
        p for p in model_dir.glob("*.dat")
        if matrix_type in p.stem.lower()
    )
    if len(matrix_files) == 0:
        raise ValueError(f"No .dat files containing '{matrix_type}' found in {model_dir}.")

    altitude_bins = au.data.load_altitude_bins(altitude_path)
    energy_bins = au.data.load_energy_bins(energy_path)

    if matrix_type == "emis":
        mats = [au.data.load_emission_matrix(p) for p in matrix_files]
        title = "Emission matrix"
    else:
        mats = [au.data.load_density_matrix(p) for p in matrix_files]
        title = "Density matrix"
    titles = tuple(p.stem for p in matrix_files)

    width, height = map(float, figsize.split(","))
    au.plot.plot_model_matrices(
        matrices=mats,
        altitude_bins=altitude_bins,
        energy_bins=energy_bins,
        titles=titles,
        title=title,
        cmap=cmap,
        figsize=(width, height),
    )
    plt.show()
