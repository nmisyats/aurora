from dataclasses import fields
from pathlib import Path

import typer
import inspect
import torch
from matplotlib import pyplot as plt

from aurora.models import MODEL_REGISTRY
from aurora.reconstruction import Reconstruction, save_reconstruction, load_reconstruction, load_static_reconstruction
from aurora.dataset import CameraRaysDataset, RadarPointsDataset
import aurora.data as data

from aurora.utils import xy_grid, xyz_grid
import aurora.plot as aplt


app = typer.Typer()


def choose_best_device(allow_gpu: bool = True):
    if allow_gpu:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def create_train_command_for_model(model_name: str, model_cls, config_cls):
    """Dynamically create a train command for a specific model"""
    
    def train_command(
        config_file: Path = typer.Argument(..., help="Path to YAML configuration file"),
        training_options: Path = typer.Option(None, help="Path to YAML training configuration file"),
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
        plot_loss: bool = typer.Option(False, help="Plot the training losses"),
        plot_flux: bool = typer.Option(True, help="Plot the reconstructed flux after training complete"),
        plot_res_x: int = typer.Option(128, help="x resolution for plotting"),
        plot_res_y: int = typer.Option(128, help="y resolution for plotting"),
        ref_flux_data: Path = typer.Option(None, help="Reference flux to compare the reconstruction with"),
        ref_flux_config: Path = typer.Option(None, help="Path to configuration file for reference flux"),
        **config_kwargs
    ):
        typer.echo(f"Training model: {model_name}")

        # Get function signature to identify default values
        sig = inspect.signature(train_command)
        
        # Create a mapping of parameter names to their default values
        param_defaults = {}
        for param_name, param in sig.parameters.items():
            if hasattr(param.default, 'default'):  # typer.Option/Argument
                param_defaults[param_name] = param.default.default
            else:
                param_defaults[param_name] = param.default

        # Load training options from YAML if provided
        train_options = {}
        if training_options is not None:
            train_options = data.load_yaml(training_options)
            typer.echo(f"Loaded training options from {training_options}")

        # Helper function to get final parameter value
        def get_param_value(param_name, current_value):
            """Get parameter value with priority: CLI args > YAML file > defaults"""
            default_value = param_defaults.get(param_name)
            yaml_value = train_options.get(param_name)
            
            # If current value differs from default, it was explicitly set via CLI
            if current_value != default_value:
                return current_value
            # Otherwise, use YAML value if available, else use current (default) value
            elif yaml_value is not None:
                return yaml_value
            else:
                return current_value

        # Apply the priority logic to all parameters
        cam_pos = get_param_value('cam_pos', cam_pos)
        cam_dir = get_param_value('cam_dir', cam_dir)
        radar_points = get_param_value('radar_points', radar_points)
        gpu = get_param_value('gpu', gpu)
        iters = get_param_value('iters', iters)
        ray_batch_size = get_param_value('ray_batch_size', ray_batch_size)
        ray_bins = get_param_value('ray_bins', ray_bins)
        radar_batch_size = get_param_value('radar_batch_size', radar_batch_size)
        ray_loss_weight = get_param_value('ray_loss_weight', ray_loss_weight)
        radar_loss_weight = get_param_value('radar_loss_weight', radar_loss_weight)
        lr = get_param_value('lr', lr)
        reg_strength = get_param_value('reg_strength', reg_strength)
        lr_step = get_param_value('lr_step', lr_step)
        lr_decay = get_param_value('lr_decay', lr_decay)
        save = get_param_value('save', save)
        plot_flux = get_param_value('plot', plot_flux)
        plot_res_x = get_param_value('plot_res_x', plot_res_x)
        plot_res_y = get_param_value('plot_res_y', plot_res_y)
        ref_flux_data = get_param_value('ref_flux_data', ref_flux_data)
        ref_flux_config = get_param_value('ref_flux_config', ref_flux_config)

        # Apply the same logic to config_kwargs (model-specific parameters)
        final_config_kwargs = {}
        for key, value in config_kwargs.items():
            final_config_kwargs[key] = get_param_value(key, value)

        # Choose device
        device = choose_best_device(gpu)

        config = data.load_config(config_file, device)
        frame = config.frame
        bbox = config.bbox
        M_emis = config.phys.emis_mat
        M_dens = config.phys.dens_mat
        E_edges = config.phys.energies
        z_edges = config.phys.altitudes

        # Load the datasets
        ray_data, radar_data = None, None
        if cam_pos is not None and cam_dir is not None:
            cams = data.load_cameras(cam_pos, cam_dir)
            ray_data = CameraRaysDataset(cams, frame, bbox)
        if radar_points is not None:
            points = data.load_radar_point_cloud(radar_points)
            radar_data = RadarPointsDataset(
                altitudes=points.altitudes,
                latitudes=points.latitudes,
                longitudes=points.longitudes,
                densities=points.densities,
                frame=frame
            )
        
        # Build config args from final kwargs
        config_args = {}
        for field_obj in fields(config_cls):
            field_name = field_obj.name
            if field_name in final_config_kwargs:
                config_args[field_name] = final_config_kwargs[field_name]
        
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
        losses = recon.train(
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
            save_reconstruction(recon, save)
        
        if plot_loss:
            aplt.plot_training_losses(*losses)

        if plot_flux:
            recon.eval_mode()
            recon_xy = xy_grid(bbox.xy_min, bbox.xy_max, plot_res_x, plot_res_y)
            estimated_f = recon.flux(recon_xy).cpu()
            rec_xy_min = bbox.xy_min.cpu()
            rec_xy_max = bbox.xy_max.cpu()
            if ref_flux_data is not None and ref_flux_config is not None:
                ref = load_static_reconstruction(ref_flux_data, ref_flux_config, device)
                reference_f = ref.flux_model.data.cpu()
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
                    flux_data=estimated_f,
                    xy_bounds=(recon.bbox.xy_min, recon.bbox.xy_max),
                    energy_edges=E_edges.cpu()
                )
        
        if plot_loss or plot_flux:
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
        for model_name, (model_cls, _) in MODEL_REGISTRY.items():
            if model_cls.__doc__ is not None:
                typer.echo(f"  {model_name}\t{model_cls.__doc__}")
            else:
                typer.echo(f"  {model_name}")
        typer.echo("\nUse 'aurora train <model_name> --help' for model-specific options.")

@train_app.command("prepare")
def make_options_file_for_model(model_name: str, file_path: Path):
    if model_name not in MODEL_REGISTRY:
        typer.echo(f"No model named {model_name}")
        raise typer.Exit(1)
    train_command = MODEL_TRAINING_COMMANDS[model_name]
    args_spec = inspect.getfullargspec(train_command)
    yaml_dict = {}
    for arg_name, arg_default in args_spec.kwonlydefaults.items():
        if arg_name == "training_options":
            continue
        annotation = args_spec.annotations[arg_name]
        if isinstance(arg_default, typer.models.OptionInfo):
            if annotation is Path:
                yaml_dict[arg_name] = "..."
            else:
                yaml_dict[arg_name] = arg_default.default
    data.save_yaml(file_path, yaml_dict)


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

@plot_app.command("flux-at", context_settings={"ignore_unknown_options": True})
def plot_flux_at(
    flux_data: Path = typer.Argument(..., help="Path to flux data"),
    config_path: Path = typer.Argument(..., help="Path to configuration YAML file"),
    x: float = typer.Argument(..., help="x coordinate"),
    y: float = typer.Argument(..., help="y coordinate"),
    gpu: bool = typer.Option(True, help="Use GPU if available"),
    plot: bool = typer.Option(True, help="Plot the generated flux"),
    save: Path = typer.Option(None, help="Save flux data to file"),
):
    """Plots the flux curve accross energy levels at a given xy location."""
    device = choose_best_device(gpu)
    recon = load_static_reconstruction(flux_data, config_path, device)
    
    xy = torch.tensor([x, y], device=device)
    f = recon.flux(xy)

    if save is not None:
        data.save_matrix_data(f.unsqueeze(1), save)
    
    if plot:
        fig, ax = aplt.plot_flux_1d(
            flux_data=f,
            energy_edges=recon.flux_model.E_edges,
            title=f"Flux at (x, y) = ({x}, {y})"
        )
        plt.show()

@plot_app.command("emis")
def plot_volume_emission(
    emis_data_path: Path = typer.Argument(..., help="Path to emission rate data"),
    config_path: Path = typer.Argument(..., help="Path to configuration YAML file"),
):
    emis_data = data.load_3d_grid_data(emis_data_path)
    config = data.load_config(config_path)
    pl = aplt.plot_volume_3d(
        volume_data=emis_data,
        xyz_bounds=(config.bbox.xyz_min, config.bbox.xyz_max),
        scalars_name="Volume emission rate"
    )
    pl.show()

@plot_app.command("dens")
def plot_electron_density():
    raise NotImplementedError

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

@gen_app.command("flux-at", context_settings={"ignore_unknown_options": True})
def generate_reconstructed_flux_at(
    recon_path: Path = typer.Argument(..., help="Path to reconstructed model"),
    x: float = typer.Argument(..., help="x coordinate"),
    y: float = typer.Argument(..., help="y coordinate"),
    gpu: bool = typer.Option(True, help="Use GPU if available"),
    plot: bool = typer.Option(True, help="Plot the generated flux"),
    save: Path = typer.Option(None, help="Save flux data to file"),
):
    """Generate the flux curve accross energy levels at a given xy location."""
    device = choose_best_device(gpu)
    recon = load_reconstruction(recon_path, device)
    recon.eval_mode()
    
    xy = torch.tensor([x, y], device=device)
    f = recon.flux(xy)

    if save is not None:
        data.save_matrix_data(f.unsqueeze(1), save)
    
    if plot:
        fig, ax = aplt.plot_flux_1d(
            flux_data=f,
            energy_edges=recon.flux_model.E_edges,
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
        ref_recon = load_static_reconstruction(recon_or_ref_path, config, device)
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
    plot: bool = typer.Option(True, help="Plot comparison"),
    cmap: str = typer.Option("viridis", help="Colormap"),
    figsize_per_col: float = typer.Option(2.0, help="Figure size per column")
):
    """Generate images using generic comparison plotting."""
    device = choose_best_device(gpu)
    
    if recon_or_ref_path.suffix == ".pth":
        recon = load_reconstruction(recon_or_ref_path, device)
    else:
        if config is None:
            typer.echo("Configuration file required for reference flux", err=True)
            raise typer.Exit(1)
        recon = load_static_reconstruction(recon_or_ref_path, config, device)
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