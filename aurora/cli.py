from dataclasses import fields
from pathlib import Path
import math

import typer
import inspect
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import matplotlib.patches as patches
import pyvista as pv
from schema import Schema, Optional, Use

from aurora.models import MODEL_REGISTRY, ReferenceFlux
from aurora.reconstruction import Reconstruction, save_reonstruction, load_reconstruction
from aurora.dataset import CameraRaysDataset, RadarPointsDataset
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.data as data
import aurora.physics as phy
from aurora.utils import (
    xy_grid,
    xyz_grid,
    bounds2d_to_tuple,
    bounds3d_to_tuple
)


app = typer.Typer()


def choose_best_device(allow_gpu: bool = True):
    if allow_gpu:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def load_reference_flux(data_path: Path, config_path: Path, device: torch.device):
    config = data.load_config(config_path, device)
    return Reconstruction(
        flux_model=ReferenceFlux(
            image=data.load_3d_grid_data(data_path).to(device),
            E_edges=config.phys.energies,
            xy_min=config.bbox.xy_min,
            xy_max=config.bbox.xy_max,
            device=device
        ),
        frame=config.frame,
        bbox=config.bbox,
        M_emis=config.phys.emis_mat,
        M_dens=config.phys.dens_mat,
        z_edges=config.phys.altitudes
    )


def create_train_command_for_model(model_name: str, model_cls, config_cls):
    """Dynamically create a train command for a specific model"""
    
    def train_command(
        origin_lat: float = typer.Argument(..., help="Latitude of the reference frame's origin"),
        origin_lon: float = typer.Argument(..., help="Longitude of the reference frame's origin"),
        origin_alt: float = typer.Argument(..., help="Altitude of the reference frame's origin"),
        field_inc: float = typer.Argument(..., help="Magnetic field inclination"),
        field_dec: float = typer.Argument(..., help="Magnetic field declination"),
        east_min: float = typer.Argument(..., help="East minimum bound for reconstruction"),
        east_max: float = typer.Argument(..., help="East maximum bound for reconstruction"),
        south_min: float = typer.Argument(..., help="South minimum bound for reconstruction"),
        south_max: float = typer.Argument(..., help="South maximum bound for reconstruction"),
        alt_min: float = typer.Argument(..., help="Altitude minimum bound for reconstruction"),
        alt_max: float = typer.Argument(..., help="Altitude maximum bound for reconstruction"),
        emis_mat: Path = typer.Argument(..., help="Emission matrix"),
        dens_mat: Path = typer.Argument(..., help="Electron density matrix"),
        altitudes: Path = typer.Argument(..., help="Altitude bins"),
        energies: Path = typer.Argument(..., help="Energy bins"),
        cam_pos: Path = typer.Option(None, help="Camera positions file"),
        cam_dir: Path = typer.Option(None, help="Cameras directory"),
        radar_points: Path = typer.Option(None, help="Radar point cloud file"),
        gpu: bool = typer.Option(True, help="Use GPU if available"),
        iters: int = typer.Option(2000, help="Number of training iterations"),
        ray_batch_size: int = typer.Option(4096, help="Batch size for ray loss"),
        ray_bins: int = typer.Option(100, help="Number of bins for ray integration"),
        radar_batch_size: int = typer.Option(1000, help="Batch size for radar loss"),
        ray_loss_weight: float = typer.Option(1.0, help="Weight for ray loss"),
        radar_loss_weight: float = typer.Option(1.0, help="Weight for radar loss"),
        lr: float = typer.Option(5e-5, help="Initial learning rate"),
        reg_strength: float = typer.Option(1.0, help="Regularization strength"),
        lr_step: int = typer.Option(1000, help="Learning rate scheduler step"),
        lr_decay: float = typer.Option(0.5, help="Learning rate step decay"),
        save: Path = typer.Option(None, help="Path to file where to save the reconstruction"),
        plot: bool = typer.Option(True, help="Plot the reconstruction after training complete"),
        plot_res_x: int = typer.Option(128, help="x resolution for plotting"),
        plot_res_y: int = typer.Option(128, help="y resolution for plotting"),
        ref_flux_data: Path = typer.Option(None, help="Reference flux to compare the reconstruction with"),
        ref_flux_config: Path = typer.Option(None, help="Path to configuration file for reference flux"),
        **config_kwargs
    ):
        typer.echo(f"Training model: {model_name}")
        
        # Choose device
        device = choose_best_device(gpu)

        # Create the oblique reference frame
        frame = Frame(
            origin_latitude=origin_lat,
            origin_longitude=origin_lon,
            origin_altitude=origin_alt,
            field_inclination=field_inc,
            field_declination=field_dec,
            device=device
        )

        # Define the reconstruction bounding box
        bbox = BBox(
            frame=frame,
            east_range=(east_min, east_max),
            south_range=(south_min, south_max),
            altitude_range=(alt_min, alt_max)
        )

        M_emis = data.load_emission_matrix(emis_mat).to(device)
        M_dens = data.load_density_matrix(dens_mat).to(device)
        E_edges = data.load_energy_bins(energies).to(device)
        z_edges = data.load_altitude_bins(altitudes).to(device)

        # Load the datasets
        ray_data, radar_data = None, None
        if cam_pos is not None and cam_dir is not None:
            cams = data.load_cameras(cam_pos, cam_dir)
            ray_data = CameraRaysDataset(cams, frame, bbox)
        if radar_points is not None:
            points = data.load_radar_point_cloud(radar_points)
            radar_data = RadarPointsDataset(*points, frame)
        
        # Build config args from kwargs
        config_args = {}
        for field_obj in fields(config_cls):
            field_name = field_obj.name
            if field_name in config_kwargs:
                config_args[field_name] = config_kwargs[field_name]
        
        # Instantiate reconstruction model
        config = config_cls(**config_args)
        f_model = model_cls(bbox.xy_min, bbox.xy_max, E_edges, config)
        f_model = f_model.to(device)
        typer.echo(f"Instantiated model:\n{f_model}")
        
        recon = Reconstruction(
            flux_model=f_model,
            frame=frame,
            bbox=bbox,
            M_emis=M_emis,
            M_dens=M_dens,
            z_edges=z_edges,
        )
        
        # Train the reconstruction on the provided data
        recon.train(
            ray_data=ray_data,
            radar_data=radar_data,
            num_iters=iters,
            ray_batch_size=ray_batch_size,
            ray_bins=ray_bins,
            radar_batch_size=radar_batch_size,
            ray_loss_weight=ray_loss_weight,
            radar_loss_weight=radar_loss_weight,
            lr=lr,
            weight_decay=reg_strength,
            lr_step_size=lr_step,
            lr_gamma=lr_decay
        )

        if save is not None:
            save_reonstruction(recon, save)

        if plot:
            recon.eval_mode()
            recon_xy = xy_grid(bbox.xy_min, bbox.xy_max, plot_res_x, plot_res_y)
            estimated_f = recon.flux(recon_xy).detach()
            estimated_q0 = phy.total_energy_flux(estimated_f, E_edges).cpu()

            x_rec_min, y_rec_min = bbox.xy_min.cpu()
            x_rec_max, y_rec_max = bbox.xy_max.cpu()

            if ref_flux_data is not None and ref_flux_config is not None:
                ref = load_reference_flux(ref_flux_data, ref_flux_config, device)

                # === Setup figure with 2 subplots and shared colorbar ===
                fig = plt.figure(figsize=(8, 4))
                grid = ImageGrid(fig, 111,
                                nrows_ncols=(1, 2),
                                axes_pad=0.1,
                                cbar_location="right",
                                cbar_mode="single",
                                cbar_size="7%",
                                cbar_pad="10%")

                # === Process reference flux ===
                reference_q0 = phy.total_energy_flux(ref.flux_model.image, E_edges).cpu()

                # === Determine color range ===
                vmin = min(estimated_q0.min(), reference_q0.min())
                vmax = max(estimated_q0.max(), reference_q0.max())

                # === Plot reference flux ===
                x_ref_min, y_ref_min = ref.bbox.xy_min.cpu()
                x_ref_max, y_ref_max = ref.bbox.xy_max.cpu()

                grid[0].imshow(reference_q0,
                            interpolation='none',
                            extent=[y_ref_min, y_ref_max, x_ref_max, x_ref_min],
                            cmap="jet",
                            vmin=vmin, vmax=vmax)

                # Highlight reconstructed area on reference plot
                rect = patches.Rectangle((y_rec_min, x_rec_min),
                                        y_rec_max - y_rec_min,
                                        x_rec_max - x_rec_min,
                                        linewidth=1, edgecolor='r', facecolor='none')
                grid[0].add_patch(rect)

                # === Plot reconstructed flux ===
                im = grid[1].imshow(estimated_q0,
                                    interpolation='none',
                                    extent=[y_rec_min, y_rec_max, x_rec_max, x_rec_min],
                                    cmap="jet",
                                    vmin=vmin, vmax=vmax)

                # === Compute and show MAE ===
                ref_f_at_xy = ref.flux(recon_xy)
                true_q0_at_grid = phy.total_energy_flux(ref_f_at_xy, E_edges).cpu()
                mae = torch.mean(torch.abs(estimated_q0 - true_q0_at_grid))
                grid[1].text(0.99, 0.01, f"MAE = {mae:.3f} mW/m$^2$",
                            transform=grid[1].transAxes,
                            ha='right', va='bottom',
                            color='white', fontsize=10,
                            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

                # === Add colorbar and labels ===
                cbar = grid[0].cax.colorbar(im)
                cbar.set_label("mW/m$^2$")

                grid[0].set_title("Reference $Q_0$")
                grid[1].set_title("Reconstructed $Q_0$")
                grid[0].set_xlabel("y (km)")
                grid[1].set_xlabel("y (km)")
                grid[0].set_ylabel("x (km)")
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(estimated_q0,
                            interpolation='none',
                            extent=[y_rec_min, y_rec_max, x_rec_max, x_rec_min],
                            cmap="jet",
                            aspect="equal")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = ax.figure.colorbar(im, cax=cax)
                cbar.set_label("mW/m$^2$")
                ax.set_xlabel("y (km)")
                ax.set_ylabel("x (km)")
                ax.set_title("$Q_0$")
                plt.show()

            plt.show()
    
    # Build parameter list dynamically
    params = []
    
    # Add fixed parameters first
    args_spec = inspect.getfullargspec(train_command)
    for arg_name, arg_default in zip(args_spec.args, args_spec.defaults):
        param_kind = inspect.Parameter.KEYWORD_ONLY
        if isinstance(arg_default, typer.models.ArgumentInfo):
            param_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        param = inspect.Parameter(arg_name, param_kind, default=arg_default, annotation=args_spec.annotations[arg_name])
        params.append(param)
    
    # Add model-specific config parameters
    for field_obj in fields(config_cls):
        field_name = field_obj.name
        field_type = field_obj.type
        
        # Handle typing annotations (e.g., Optional[int], Union types, etc.)
        origin_type = getattr(field_type, '__origin__', None)
        if origin_type is not None:
            # For Optional[T], Union[T, None], etc., get the first non-None type
            args = getattr(field_type, '__args__', ())
            field_type = next((arg for arg in args if arg is not type(None)), field_type)
        
        # Get default value
        if field_obj.default is not inspect._empty:
            default_val = field_obj.default
        elif field_obj.default_factory is not inspect._empty:
            default_val = field_obj.default_factory()
        else:
            default_val = ...  # Required parameter
        
        # Get help text
        help_text = field_obj.metadata.get("help", f"Config parameter: {field_name}")
        
        # Create typer option with proper type annotation
        param_default = typer.Option(default_val, help=help_text)
        
        params.append(
            inspect.Parameter(
                field_name, 
                inspect.Parameter.KEYWORD_ONLY, 
                default=param_default,
                annotation=field_type  # This is crucial for type conversion
            )
        )
    
    # Set the new signature
    train_command.__signature__ = inspect.Signature(params)
    train_command.__name__ = f"train_{model_name}"
    
    return train_command

# Create subcommand for train
train_app = typer.Typer(help="Train models")
app.add_typer(train_app, name="train")

# Dynamically register all models as subcommands
MODEL_TRAINING_COMMANDS = {}
for model_name, model_entry in MODEL_REGISTRY.items():
    model_cls, config_cls = model_entry
    train_func = create_train_command_for_model(model_name, model_cls, config_cls)
    command = train_app.command(name=model_name, help=f"Train {model_name} model")
    train_command = command(train_func)
    MODEL_TRAINING_COMMANDS[model_name] = train_command

# Fallback command to list available models
@train_app.callback(invoke_without_command=True)
def train_main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo("Available models:")
        for model_name in MODEL_REGISTRY.keys():
            typer.echo(f"  {model_name}")
        typer.echo("\nUse 'aurora train <model_name> --help' for model-specific options.")

def get_full_args_with_defaults(func, config_dict):
    sig = inspect.signature(func)
    args = {}

    for name, param in sig.parameters.items():
        if name in config_dict:
            args[name] = config_dict[name]
        else:
            default = param.default
            if isinstance(default, typer.models.OptionInfo):
                args[name] = default.default  # actual default
            elif default is not inspect.Parameter.empty:
                args[name] = default
            else:
                raise ValueError(f"Missing required parameter: {name}")
    return args

def create_train_from_config_command_for_model(model_name: str, train_func):
    args_spec = inspect.signature(train_func)
    params = []

    # Add config_path argument
    params.append(inspect.Parameter(
        "config_path",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=Path,
        default=...,
    ))
    # Add training_path argument
    params.append(inspect.Parameter(
        "training_path",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=Path,
        default=...,
    ))

    # For every parameter in train_func, add a matching Option to override
    for name, param in args_spec.parameters.items():
        if name == "self":
            continue

        # We skip config_path — already added
        if name == "config_path":
            continue
        
        annotation = param.annotation
        default_val = param.default

        if isinstance(default_val, typer.models.ArgumentInfo) or isinstance(default_val, typer.models.OptionInfo):
            # Extract help text
            help_text = default_val.help

            # Treat everything as Option with default None (means: override optional)
            default = typer.Option(None, help=help_text)
        elif default_val != inspect._empty:
            default = typer.Option(default_val)
        else:
            # No default known — make it explicitly optional
            default = typer.Option(None)

        params.append(inspect.Parameter(
            name,
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=default,
            annotation=annotation
        ))

    # Create a wrapper function
    def from_config_wrapper(config_path: Path, training_path: Path, **cli_kwargs):
        config_schema = data.config_schema(config_path.parent)

        training_schema = {}
        for param in params:
            k = Optional(param.name)
            if param.annotation is Path:
                training_schema[k] = data.path_validator(training_path.parent)
            else:
                training_schema[k] = Use(param.annotation)
        training_schema = Schema(training_schema)

        config_dict = data.load_yaml(config_path, config_schema)
        training_dict = data.load_yaml(training_path, training_schema)

        merged = {}
        for v in config_dict.values():
            merged.update(v) # Expend inner configuration blocks
        # Add cli kwargs overrides
        merged.update({k: v for k, v in cli_kwargs.items() if v is not None})
        merged.update(training_dict)

        args = get_full_args_with_defaults(train_func, merged)
        return train_func(**args)
    
    from_config_wrapper.__signature__ = inspect.Signature(params)
    from_config_wrapper.__name__ = f"train_from_config_{model_name}"
    return from_config_wrapper

from_config_app = typer.Typer(help="Train models using config file + CLI override")
train_app.add_typer(from_config_app, name="from-config")

# Dynamically register all from-config commands
for model_name, train_func in MODEL_TRAINING_COMMANDS.items():
    from_config_func = create_train_from_config_command_for_model(model_name, train_func)
    command = from_config_app.command(name=model_name, help=f"Train {model_name} using a config file and CLI overrides")
    command(from_config_func)


# Create subcommand for plotting
plot_app = typer.Typer(help="Plotting utilities")
app.add_typer(plot_app, name="plot")

@plot_app.command("flux")
def plot_flux(
    flux_data: Path = typer.Argument(..., help="Path to flux data"),
    config_path: Path = typer.Argument(..., help="Path to configuration YAML file")
):
    f_image = data.load_3d_grid_data(flux_data)
    config = data.load_config(config_path)
    
    xy_min = config.bbox.xy_min
    xy_max = config.bbox.xy_max
    q0 = phy.total_energy_flux(f_image)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    x_min, x_max, y_min, y_max = bounds2d_to_tuple(xy_min, xy_max)
    im = ax.imshow(q0.cpu(),
                interpolation='none',
                extent=[y_min, y_max, x_max, x_min],
                cmap="jet",
                aspect="equal")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("mW/m$^2$")
    ax.set_xlabel("y (km)")
    ax.set_ylabel("x (km)")
    ax.set_title("$Q_0$")
    plt.show()

@plot_app.command("cams")
def plot_cameras(
    cam_pos: Path = typer.Argument(..., help="Camera positions file"),
    cam_dir: Path = typer.Argument(..., help="Cameras directory"),
    location: str = typer.Option(None, help="Name of the camera to plot")
):
    cams = data.load_cameras(cam_pos, cam_dir)

    if location is not None:
        cams = {cam.name: cam for cam in cams}
        cam = cams[location]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cam.image)
        cbar = ax.figure.colorbar(im)
        cbar.set_label("Rayleigh")
        ax.set_title(f"{location} ({cam.latitude:.3f}°N {cam.longitude:.3f}°E +{cam.altitude:.3f}km)")
        plt.show()
    else:
        n_imgs = len(cams)
        # Compute grid size (try to make it as square as possible)
        cols = math.ceil(math.sqrt(n_imgs))
        rows = math.ceil(n_imgs / cols)
        # Find global min/max for consistent color scaling
        vmin = min(cam.image.min() for cam in cams)
        vmax = max(cam.image.max() for cam in cams)
        # Create figure and image grid
        fig = plt.figure(figsize=(cols * 3, rows * 3))
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(rows, cols),
                        axes_pad=0.4,
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="single",
                        cbar_size="5%",
                        cbar_pad=0.1)
        # Plot images
        for ax, cam in zip(grid, cams):
            im = ax.imshow(cam.image, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.axis('off')
            ax.set_title(cam.name)
        # Turn off unused axes
        for ax in grid[n_imgs:]:
            ax.axis('off')
        # Shared colorbar
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.set_label("Rayleigh")
        plt.show()


# Create subcommand for generating
gen_app = typer.Typer(help="Generation utilities")
app.add_typer(gen_app, name="gen")

@gen_app.command("flux")
def generate_reconstructed_flux(
    recon_path: Path = typer.Argument(..., help="Path to reconstructed model"),
    res_x: int = typer.Option(128),
    res_y: int = typer.Option(128),
    gpu: bool = typer.Option(True),
    plot: bool = typer.Option(True),
    save: Path = typer.Option(None)
):
    device = choose_best_device(gpu)
    recon = load_reconstruction(recon_path, device)
    recon.eval_mode()
    xy_min = recon.bbox.xy_min
    xy_max = recon.bbox.xy_max
    xy = xy_grid(xy_min, xy_max, res_x, res_y)
    f = recon.flux(xy)

    if save is not None:
        data.save_3d_grid_data(f, save)
    
    if plot:
        q0 = phy.total_energy_flux(f, recon.flux_model.E_edges)
        fig, ax = plt.subplots(figsize=(8, 6))
        x_min, x_max, y_min, y_max = bounds2d_to_tuple(xy_min, xy_max)
        im = ax.imshow(q0.cpu(),
                    interpolation='none',
                    extent=[y_min, y_max, x_max, x_min],
                    cmap="jet",
                    aspect="equal")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = ax.figure.colorbar(im, cax=cax)
        cbar.set_label("mW/m$^2$")
        ax.set_xlabel("y (km)")
        ax.set_ylabel("x (km)")
        ax.set_title("$Q_0$")
        plt.show()

@gen_app.command("emis")
def generate_volume_emission(
    recon_or_ref_path: Path = typer.Argument(..., help="Path to reconstruction or reference flux"),
    config: Path = typer.Option(None, help="Path to configuration for reference flux"),
    res_x: int = typer.Option(100),
    res_y: int = typer.Option(100),
    res_z: int = typer.Option(50),
    gpu: bool = typer.Option(True),
    plot: bool = typer.Option(True),
    save: Path = typer.Option(None)
):
    device = choose_best_device(gpu)
    if recon_or_ref_path.suffix == ".pth":
        recon = load_reconstruction(recon_or_ref_path, device)
        recon.eval_mode()
        xyz_min = recon.bbox.xyz_min
        xyz_max = recon.bbox.xyz_max
        xyz = xyz_grid(xyz_min, xyz_max, res_x, res_y, res_z)
        l = recon.emis_rate(xyz).cpu()
    else:
        if config is None:
            typer.echo("Configuration file required for reference flux", err=True)
            raise typer.Exit(1)
        ref_recon = load_reference_flux(recon_or_ref_path, config, device)
        xyz_min = ref_recon.bbox.xyz_min
        xyz_max = ref_recon.bbox.xyz_max
        xyz = xyz_grid(xyz_min, xyz_max, res_x, res_y, res_z)
        l = ref_recon.emis_rate(xyz).cpu()
    
    if save is not None:
        data.save_3d_grid_data(l, save)
    
    if plot:
        l = l.numpy()
        grid = pv.ImageData()
        grid.dimensions = (res_x+1, res_y+1, res_z+1)  # Add 1 because dimensions are number of points
        x_min, x_max, y_min, y_max, z_min, z_max = bounds3d_to_tuple(xyz_min, xyz_max)
        x_scale = (x_max - x_min) / (z_max - z_min)
        y_scale = (y_max - y_min) / (z_max - z_min)
        z_scale = 1
        grid.spacing = (x_scale, y_scale, z_scale)  # Voxel spacing
        grid.origin = (0, 0, 0)   # Origin of the grid
        # Add the density data to the grid as a cell array
        # Need to flatten the numpy array to match PyVista's expected format
        grid.cell_data["density"] = l.flatten(order="F")
        # Create a custom opacity transfer function
        # This maps density values to opacity
        opacity = [0, 0.1, 0.3, 0.6, 0.8, 1.0, 1.0]
        # Create the plotter
        pl = pv.Plotter()
        # Add the volume to the plotter with a colormap
        # pl.add_volume(grid, scalars="density", cmap="viridis", opacity=opacity, shade=False)
        pl.add_volume(grid, scalars="density", cmap="coolwarm", opacity=opacity, shade=False)
        # Optional: Add axes for reference
        pl.show_axes()
        pl.add_bounding_box()
        # Display the plot
        pl.show()

@gen_app.command("imgs")
def generate_images(
    recon_or_ref_path: Path = typer.Argument(..., help="Path to reconstruction"),
    cam_pos: Path = typer.Argument(..., help="Camera positions file"),
    cam_dir: Path = typer.Argument(..., help="Cameras directory"),
    config: Path = typer.Option(None, help="Path to configuration for reference flux"),
    locations: str = typer.Option(None, parser=lambda s: s.split(",")),
    ray_bins: int = typer.Option(100),
    downsample: int = typer.Option(None),
    gpu: bool = typer.Option(True),
    plot: bool = typer.Option(True),
):
    device = choose_best_device(gpu)
    if recon_or_ref_path.suffix == ".pth":
        device = choose_best_device(gpu)
        recon = load_reconstruction(recon_or_ref_path, device)
    else:
        recon = load_reference_flux(recon_or_ref_path, config, device)
    recon.eval_mode()

    cams = data.load_cameras(cam_pos, cam_dir)
    if locations is not None:
        cams = list(filter(lambda c: c.name in locations, cams))
    n_cam = len(cams)

    if downsample is not None:
        for i in range(n_cam):
            cams[i] = cams[i].downsample(downsample)

    imgs = []
    for i, cam in enumerate(cams):
        print(f"Generating image {i+1}/{n_cam} ({cam.name})")
        img = recon.image(cam, ray_bins)
        imgs.append(img)

    if plot:
        # fig, axs = plt.subplots(2, n_img, figsize=(n_img * 2, 4 + 0.5))  # Added extra space for colorbar
        # plt.subplots_adjust(wspace=0.05, hspace=0.05)
        fig = plt.figure(figsize=(n_cam * 2 + 3.0, 4 + 0.5))
        # Lists to store min and max values for color scaling
        all_mins, all_maxs = [], []
        refs = []
        # First pass to get min and max values across all images
        for i in range(n_cam):
            imgs[i] = imgs[i].numpy(force=True)
            refs.append(cams[i].image)
            all_mins.append(min(imgs[i].min(), refs[i].min()))
            all_maxs.append(max(imgs[i].max(), refs[i].max()))
        # Get global min and max
        vmin = min(all_mins)
        vmax = max(all_maxs)
        # Second pass to plot images with consistent color scale
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(2, n_cam),
                        axes_pad=0.1,
                        cbar_location="right", cbar_mode="single", cbar_size="7%", cbar_pad="10%")
        cat = [*imgs, *refs]
        ims = []
        for i, (ax, img) in enumerate(zip(grid, cat)):
            im = ax.imshow(img, vmin=vmin, vmax=vmax)
            ims.append(im)
            ax.set_xticks([])
            ax.set_yticks([])
            if i < n_cam:
                ax.set_title(cams[i].name)
        cbar = grid[0].cax.colorbar(ims[0])
        cbar.set_label("Rayleigh")
        grid[0].set_ylabel("Generated image")
        grid[n_cam].set_ylabel("Reference image")
        plt.show()