from dataclasses import fields
from pathlib import Path

import typer
import inspect
import torch
from matplotlib import pyplot as plt
from schema import Schema, Optional, Use

from aurora.models import MODEL_REGISTRY, ReferenceFlux
from aurora.reconstruction import Reconstruction, save_reonstruction, load_reconstruction
from aurora.dataset import CameraRaysDataset, RadarPointsDataset
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.data as data
import aurora.physics as phy
from aurora.utils import xy_grid, xyz_grid
import aurora.plot as aplt


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
            estimated_f = recon.flux(recon_xy).cpu()
            rec_xy_min = bbox.xy_min.cpu()
            rec_xy_max = bbox.xy_max.cpu()
            if ref_flux_data is not None and ref_flux_config is not None:
                ref = load_reference_flux(ref_flux_data, ref_flux_config, device)
                reference_f = ref.flux_model.image.cpu()
                ref_xy_min = ref.bbox.xy_min.cpu()
                ref_xy_max = ref.bbox.xy_max.cpu()
                aplt.plot_flux_2d_comparison(
                    estimated_flux=estimated_f,
                    reference_flux=reference_f,
                    estimated_bounds=(rec_xy_min, rec_xy_max),
                    reference_bounds=(ref_xy_min, ref_xy_max),
                    energy_edges=E_edges.cpu(),
                )
            else:
                aplt.plot_flux_2d(
                    flux_data=reference_f,
                    xy_bounds=(recon.bbox.xy_min, recon.bbox.xy_max),
                )
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
        "options_file",
        kind=inspect.Parameter.KEYWORD_ONLY,
        default=typer.Option(None, help="YAML file containing training options and overrides"),
        annotation=Path
    ))

    # For every parameter in train_func, add a matching Option to override
    for name, param in args_spec.parameters.items():
        if name in ("self", "config_path", "options_file"):
            continue

        annotation = param.annotation
        default_val = param.default

        if isinstance(default_val, typer.models.ArgumentInfo):
            continue

        if isinstance(default_val, typer.models.OptionInfo):
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
    def from_config_wrapper(config_path: Path, options_file: Path, **cli_kwargs):
        config_schema = data.config_schema(config_path.parent)
        config_dict = data.load_yaml(config_path, config_schema)
        
        training_dict = {}

        if options_file is not None:
            training_schema = {}
            for param in params:
                k = Optional(param.name)
                if param.annotation is Path:
                    training_schema[k] = data.path_validator(options_file.parent)
                else:
                    training_schema[k] = Use(param.annotation)
            training_schema = Schema(training_schema)
            training_dict = data.load_yaml(options_file, training_schema)

        merged = {}
        for v in config_dict.values():
            merged.update(v) # Expend inner configuration blocks
        # Add kwargs from training file
        merged.update(training_dict)
        # Add cli kwargs overrides
        merged.update({k: v for k, v in cli_kwargs.items() if v is not None})

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
    config_path: Path = typer.Argument(..., help="Path to configuration YAML file"),
    title: str = typer.Option("$Q_0$", help="Plot title"),
    cmap: str = typer.Option("jet", help="Colormap name"),
    figsize: str = typer.Option("8,6", help="Figure size as 'width,height'")
):
    """Plot flux data using the generic plotting function."""
    f_image = data.load_3d_grid_data(flux_data)
    config = data.load_config(config_path)
    
    # Parse figsize
    width, height = map(float, figsize.split(','))
    
    # Use the generic plotting function
    fig, ax = aplt.plot_flux_2d(
        flux_data=f_image,
        xy_bounds=(config.bbox.xy_min, config.bbox.xy_max),
        energy_edges=config.phys.energies,
        title=title,
        cmap=cmap,
        figsize=(width, height)
    )
    plt.show()

@plot_app.command("cams")
def plot_cameras(
    cam_pos: Path = typer.Argument(..., help="Camera positions file"),
    cam_dir: Path = typer.Argument(..., help="Cameras directory"),
    location: str = typer.Option(None, help="Name of specific camera to plot"),
    title_format: str = typer.Option("{name} ({latitude:.3f}°N {longitude:.3f}°E +{altitude:.3f}km)", help="Title format"),
    cmap: str = typer.Option("viridis", help="Colormap name"),
    figsize_per_image: float = typer.Option(3.0, help="Size factor per image"),
    max_cols: int = typer.Option(None, help="Maximum columns in grid")
):
    """Plot camera images using the generic plotting function."""
    cams = data.load_cameras(cam_pos, cam_dir)

    fig, axes = aplt.plot_cameras_grid(
        cameras=cams,
        selected_camera=location,
        figsize_per_image=figsize_per_image,
        cmap=cmap,
        title_format=title_format,
        max_cols=max_cols
    )
    plt.show()


# Create subcommand for generating
gen_app = typer.Typer(help="Generation utilities")
app.add_typer(gen_app, name="gen")

@gen_app.command("flux")
def generate_reconstructed_flux(
    recon_path: Path = typer.Argument(..., help="Path to reconstructed model"),
    res_x: int = typer.Option(128, help="X resolution"),
    res_y: int = typer.Option(128, help="Y resolution"),
    gpu: bool = typer.Option(True, help="Use GPU if available"),
    plot: bool = typer.Option(True, help="Plot the generated flux"),
    save: Path = typer.Option(None, help="Save flux data to file"),
    cmap: str = typer.Option("jet", help="Colormap for plotting")
):
    """Generate flux from reconstruction using generic plotting."""
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
        fig, ax = aplt.plot_flux_2d(
            flux_data=f,
            xy_bounds=(xy_min, xy_max),
            energy_edges=recon.flux_model.E_edges,
            cmap=cmap
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
    plot: bool = typer.Option(True, help="Plot the volume"),
    save: Path = typer.Option(None, help="Save volume data"),
    cmap: str = typer.Option("coolwarm", help="Volume colormap"),
    opacity: str = typer.Option("0,0.1,0.3,0.6,0.8,1.0,1.0", help="Opacity values as comma-separated list")
):
    """Generate volume emission using generic 3D plotting."""
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
        # Parse opacity values
        opacity_vals = list(map(float, opacity.split(',')))
        
        plotter = aplt.plot_volume_3d(
            volume_data=l,
            xyz_bounds=(xyz_min, xyz_max),
            cmap=cmap,
            opacity=opacity_vals
        )
        plotter.show()

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
    plot: bool = typer.Option(True, help="Plot comparison"),
    cmap: str = typer.Option("viridis", help="Colormap"),
    figsize_per_col: float = typer.Option(2.0, help="Figure size per column")
):
    """Generate images using generic comparison plotting."""
    device = choose_best_device(gpu)
    
    if recon_or_ref_path.suffix == ".pth":
        recon = load_reconstruction(recon_or_ref_path, device)
    else:
        recon = load_reference_flux(recon_or_ref_path, config, device)
    recon.eval_mode()

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
        img = recon.image(cam, ray_bins)
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