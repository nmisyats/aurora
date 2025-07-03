from pathlib import Path
import inspect
from typing import Callable, Tuple, Dict, List

import typer
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

import aurora as au
from aurora import models, data
from aurora.samplers import StratifiedSampler
from aurora.utils import choose_best_device, xy_grid
from aurora.models import FluxModel
import aurora.plot as aplt


# Create subcommand for train
train_app = typer.Typer(help="Train models")

# Dynamically register all models as subcommands
MODEL_REGISTRY = {}
MODEL_TRAINING_COMMANDS = {}
    

def model_train_command(model_name: str):
    """Dynamically create a train command for a specific model"""
    
    def make_train_func(train_model: Callable[..., Tuple[FluxModel, Dict[str, List[float]]]]):
        def train_func(
            config_file: Path = typer.Argument(..., help="Path to YAML configuration file"),
            options: Path = typer.Option(None, help="Path to YAML training configuration file"),
            cam_pos: Path = typer.Option(None, help="Camera positions file"),
            cam_dir: Path = typer.Option(None, help="Cameras directory"),
            radar: Path = typer.Option(None, help="Radar point cloud file"),
            gpu: bool = typer.Option(True, help="Use GPU if available"),
            iters: int = typer.Option(2000, help="Number of training iterations"),
            ray_batch: int = typer.Option(4096, help="Batch size for ray loss"),
            ray_bins: int = typer.Option(100, help="Number of bins for ray integration"),
            radar_batch: int = typer.Option(1024, help="Batch size for radar loss"),
            smooth_batch: int = typer.Option(1024, help="Batch size for spectral smoothness loss"),
            ray_weight: float = typer.Option(1.0, help="Weight for ray loss"),
            radar_weight: float = typer.Option(1.0, help="Weight for radar loss"),
            smooth_weight: float = typer.Option(0.0, help="Weight for spectral smoothness loss"),
            lr: float = typer.Option(5e-5, help="Initial learning rate"),
            reg_strength: float = typer.Option(1.0, help="Parameter L2 regularization strength"),
            lr_step: int = typer.Option(1000, help="Learning rate scheduler step"),
            lr_decay: float = typer.Option(0.5, help="Learning rate step decay"),
            save: Path = typer.Option(None, help="Path to file where to save the reconstruction"),
            plot_loss: bool = typer.Option(False, help="Plot the training losses"),
            plot_flux: bool = typer.Option(True, help="Plot the reconstructed flux after training complete"),
            plot_res_x: int = typer.Option(128, help="x resolution for plotting"),
            plot_res_y: int = typer.Option(128, help="y resolution for plotting"),
            ref_flux: Path = typer.Option(None, help="Reference flux to compare the reconstruction with"),
            ref_config: Path = typer.Option(None, help="Path to configuration file for reference flux"),
            **model_kwargs
        ):
            typer.echo(f"Training model: {model_name}")

            # Get function signature to identify default values
            sig = inspect.signature(train_func)
            
            # Create a mapping of parameter names to their default values
            param_defaults = {}
            for param_name, param in sig.parameters.items():
                if hasattr(param.default, 'default'):  # typer.Option/Argument
                    param_defaults[param_name] = param.default.default
                else:
                    param_defaults[param_name] = param.default

            # Load training options from YAML if provided
            train_options = {}
            if options is not None:
                train_options = data.load_yaml(options)
                typer.echo(f"Loaded training options from {options}")

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
            radar = get_param_value('radar', radar)
            gpu = get_param_value('gpu', gpu)
            iters = get_param_value('iters', iters)
            ray_batch = get_param_value('ray_batch', ray_batch)
            ray_bins = get_param_value('ray_bins', ray_bins)
            radar_batch = get_param_value('radar_batch', radar_batch)
            smooth_batch = get_param_value('smooth_batch', smooth_batch)
            ray_weight = get_param_value('ray_weight', ray_weight)
            radar_weight = get_param_value('radar_weight', radar_weight)
            smooth_weight = get_param_value('smooth_weight', smooth_weight)
            lr = get_param_value('lr', lr)
            reg_strength = get_param_value('reg_strength', reg_strength)
            lr_step = get_param_value('lr_step', lr_step)
            lr_decay = get_param_value('lr_decay', lr_decay)
            save = get_param_value('save', save)
            plot_flux = get_param_value('plot', plot_flux)
            plot_res_x = get_param_value('plot_res_x', plot_res_x)
            plot_res_y = get_param_value('plot_res_y', plot_res_y)
            ref_flux = get_param_value('ref_flux', ref_flux)
            ref_config = get_param_value('ref_config', ref_config)

            # Apply the same logic to config_kwargs (model-specific parameters)
            final_config_kwargs = {}
            for key, value in model_kwargs.items():
                final_config_kwargs[key] = get_param_value(key, value)

            # Choose device
            device = choose_best_device(gpu)

            config = data.load_config(config_file, device)
            frame = config.frame
            bbox = config.bbox
            emis_mat = config.physics.emis_mat
            dens_mat = config.physics.dens_mat
            energy_bins = config.physics.energy_bins
            altitude_bins = config.physics.altitude_bins

            # Load the datasets
            ray_data, radar_data = None, None
            if cam_pos is not None and cam_dir is not None:
                cams = data.load_cameras(cam_pos, cam_dir)
                ray_data = au.datasets.RayDataset(cams, frame, bbox)
            if radar is not None:
                points = data.load_radar_point_cloud(radar)
                radar_data = au.datasets.RadarDataset(
                    altitudes=points.altitudes,
                    latitudes=points.latitudes,
                    longitudes=points.longitudes,
                    densities=points.densities,
                    frame=frame
                )
            
            # Run model specific training
            model, history = train_model(**locals(), **final_config_kwargs)

            if save is not None:
                models.save_model(model, save)
                print(f"Saved model in {save}")
            
            if plot_loss:
                aplt.plot_training_losses(history)

            if plot_flux:
                model.eval()
                recon_xy = xy_grid(bbox.xy_min, bbox.xy_max, plot_res_x, plot_res_y)
                estimated_f = model.flux(recon_xy)
                rec_xy_min = bbox.xy_min
                rec_xy_max = bbox.xy_max
                if ref_flux is not None and ref_config is not None:
                    ref = models.load_grid_model(ref_flux, ref_config, device)
                    reference_f = ref.data
                    ref_xy_min = ref.bbox.xy_min
                    ref_xy_max = ref.bbox.xy_max
                    aplt.plot_flux_2d_comparison(
                        estimated_flux=estimated_f,
                        reference_flux=reference_f,
                        estimated_bounds=(rec_xy_min, rec_xy_max),
                        reference_bounds=(ref_xy_min, ref_xy_max),
                        energy_edges=energy_bins,
                    )
                else:
                    aplt.plot_flux_2d(
                        flux_data=estimated_f,
                        xy_bounds=(model.bbox.xy_min, model.bbox.xy_max),
                        energy_edges=energy_bins
                    )
            
            if plot_loss or plot_flux:
                plt.show()
        
        # Build parameter list dynamically
        params = []
        
        # Add fixed parameters first
        args_spec = inspect.getfullargspec(train_func)
        for arg_name, arg_default in zip(args_spec.args, args_spec.defaults):
            param_kind = inspect.Parameter.KEYWORD_ONLY
            if isinstance(arg_default, typer.models.ArgumentInfo):
                param_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
            param = inspect.Parameter(arg_name, param_kind, default=arg_default, annotation=args_spec.annotations[arg_name])
            params.append(param)
        
        # Add model-specific config parameters
        args_spec = inspect.getfullargspec(train_model)
        defaults_start_index = len(args_spec.args) - len(args_spec.defaults)
        for arg_name in args_spec.args:
            arg_type = args_spec.annotations[arg_name]

            arg_index = args_spec.args.index(arg_name)
            arg_default_index = arg_index - defaults_start_index
            arg_default = args_spec.defaults[arg_default_index]

            # Handle typing annotations (e.g., Optional[int], Union types, etc.)
            origin_type = getattr(arg_type, '__origin__', None)
            if origin_type is not None:
                # For Optional[T], Union[T, None], etc., get the first non-None type
                args = getattr(arg_type, '__args__', ())
                arg_type = next((arg for arg in args if arg is not type(None)), arg_type)
            
            # Create typer option with proper type annotation
            param_default = arg_default
            
            params.append(
                inspect.Parameter(
                    arg_name, 
                    inspect.Parameter.KEYWORD_ONLY, 
                    default=param_default,
                    annotation=arg_type  # This is crucial for type conversion
                )
            )
        
        # Set the new signature
        train_func.__signature__ = inspect.Signature(params)
        train_func.__name__ = f"train_{model_name}"

        command = train_app.command(name=model_name, help=f"Train {model_name} model")
        command(train_func)
    
    return make_train_func

    # MODEL_REGISTRY[model_name] = model_cls
    # MODEL_TRAINING_COMMANDS[model_name] = train_command

def default_iter_loss(**kw):
    def iter_loss(model: FluxModel):
        # Train the reconstruction on the provided data
        loss_dict = {}
        total_loss = torch.scalar_tensor(0.0, device=kw["device"])
        if kw["ray_data"] is not None:
            ray_batch = kw["ray_data"].sample_batch(kw["ray_batch"])
            ray_sampler = StratifiedSampler(kw["ray_bins"])
            ray_loss = kw["ray_weight"] * au.ray_loss(model, ray_batch, ray_sampler)
            total_loss = total_loss + ray_loss
            loss_dict["ray_loss"] = ray_loss.item()
        if kw["radar_data"] is not None:
            radar_batch = kw["radar_data"].sample_batch(kw["radar_batch"])
            radar_loss = kw["radar_weight"] * au.radar_loss(model, radar_batch)
            total_loss = total_loss + radar_loss
            loss_dict["radar_loss"] = radar_loss.item()
        if kw["smooth_weight"] > 0.0:
            xy_norm = torch.rand(kw["smooth_batch"], 2, device=kw["device"])
            xy = model.bbox.real_xy(xy_norm)
            smooth_loss = kw["smooth_weight"] * au.spectral_smoothness_loss(model, xy)
            loss_dict["smooth_loss"] = smooth_loss.item()
        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
    return iter_loss

@model_train_command("spectral_mlp")
def train_spectral_mlp(
    enc_exp: int = typer.Option(4, help="Maximum positional encoding exponent"),
    max_log_f: int = typer.Option(7.0, help="Maximum logarithmic value of the reconstructed flux"),
    num_hidden: int = typer.Option(4, help="Number of hidden layers"),
    hidden_size: int = typer.Option(128, help="Size of each hidden layer"),
    **kw
):
    # Instantiate reconstruction model
    model = models.SpectralMLP(
        frame=kw["frame"],
        bbox=kw["bbox"],
        emis_mat=kw["emis_mat"],
        dens_mat=kw["dens_mat"],
        altitude_bins=kw["altitude_bins"],
        energy_bins=kw["energy_bins"],
        encoding_exp=enc_exp,
        max_log_flux=max_log_f,
        num_hidden=num_hidden,
        hidden_size=hidden_size
    ).to(kw["device"])
    typer.echo(f"Instantiated model:\n{model}")
    # Train the reconstruction on the provided data
    history = au.train(
        model=model,
        iter_loss=default_iter_loss(**kw),
        num_iters=kw["iters"],
        lr=kw["lr"],
        weight_decay=kw["reg_strength"],
    )
    return model, history


# Fallback command to list available models
# @train_app.callback(invoke_without_command=True)
# def train_main(ctx: typer.Context):
#     if ctx.invoked_subcommand is None:
#         typer.echo("Available models:")
#         for model_name, model_cls in MODEL_REGISTRY.items():
#             if model_cls.__doc__ is not None:
#                 typer.echo(f"  {model_name} - {model_cls.__doc__}")
#             else:
#                 typer.echo(f"  {model_name}")
#         typer.echo("\nUse 'aurora train <model_name> --help' for model-specific options.")

# @train_app.command("prepare")
# def make_options_file_for_model(model_name: str, file_path: Path):
#     """Generate a template training options YAML for the specified model"""
#     if model_name not in MODEL_TRAINING_COMMANDS:
#         typer.echo(f"No model named {model_name}")
#         raise typer.Exit(1)
#     train_command = MODEL_TRAINING_COMMANDS[model_name]
#     args_spec = inspect.getfullargspec(train_command)
#     yaml_dict = {}
#     for arg_name, arg_default in args_spec.kwonlydefaults.items():
#         if arg_name == "options":
#             continue
#         annotation = args_spec.annotations[arg_name]
#         if isinstance(arg_default, typer.models.OptionInfo):
#             if annotation is Path:
#                 yaml_dict[arg_name] = "..."
#             else:
#                 yaml_dict[arg_name] = arg_default.default
#     data.save_yaml(file_path, yaml_dict)
