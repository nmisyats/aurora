import typer
from dataclasses import fields
import torch
import inspect
from pathlib import Path

from aurora.models import MODEL_REGISTRY
from aurora.reconstruction import Reconstruction, RayDataset
from aurora.data import load_physical_model, load_cameras
from aurora.plot import plot_total_energy_flux, plot_model_matrix

app = typer.Typer()

def create_train_command_for_model(model_name: str, model_cls, config_cls):
    """Dynamically create a train command for a specific model"""
    
    def train_command(
        physical_model_path: Path = typer.Argument(..., help="Path to physical model configuration file"),
        camera_dataset_path: Path = typer.Argument(..., help="Path to camera dataset configuration file"),
        gpu: bool = typer.Option(True, help="Use GPU if available"),
        iters: int = typer.Option(2000, help="Number of training iterations"),
        batch_size: int = typer.Option(4096, help="Batch size"),
        ray_bins: int = typer.Option(100, help="Number of bins for ray integration"),
        lr: float = typer.Option(5e-5, help="Initial learning rate"),
        reg_strength: float = typer.Option(1.0, help="Regularization strength"),
        lr_step: int = typer.Option(1000, help="Learning rate scheduler step"),
        lr_decay: float = typer.Option(0.5, help="Learning rate step decay"),
        plot: bool = typer.Option(False, help="Plot the reconstruction after training complete"),
        **config_kwargs
    ):
        typer.echo(f"Training model: {model_name}")
        
        # Choose device
        if gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        
        # Load physics model
        pm, f_ref = load_physical_model(physical_model_path, device)
        
        # Build config args from kwargs
        config_args = {}
        for field_obj in fields(config_cls):
            field_name = field_obj.name
            if field_name in config_kwargs:
                config_args[field_name] = config_kwargs[field_name]
        
        # Instantiate reconstruction model
        config = config_cls(**config_args)
        f_model = model_cls(pm, config)
        f_model = f_model.to(device)
        typer.echo(f"Instantiated model:\n{f_model}")
        
        recon = Reconstruction(pm, f_model, device)
        
        # Train the reconstruction after loading the camera dataset
        cams = load_cameras(camera_dataset_path)
        dataset = RayDataset(cams, pm.frame, device)
        recon.train(
            dataset=dataset,
            num_iters=iters,
            batch_size=batch_size,
            ray_bins=ray_bins,
            lr=lr,
            weight_decay=reg_strength,
            lr_step_size=lr_step,
            lr_gamma=lr_decay
        )

        if plot:
            recon.eval_mode()
            plot_total_energy_flux(recon.f, pm, 128, 128, f_ref)
    
    # Build parameter list dynamically
    params = []
    
    # Add fixed parameters first
    args_spec = inspect.getfullargspec(train_command)
    for arg_name, arg_default in zip(args_spec.args, args_spec.defaults):
        param_kind = inspect.Parameter.KEYWORD_ONLY
        if isinstance(arg_default, typer.models.ArgumentInfo):
            param_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        param = inspect.Parameter(arg_name, param_kind, default=arg_default)
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
for model_name, model_entry in MODEL_REGISTRY.items():
    model_cls, config_cls = model_entry
    train_command = create_train_command_for_model(model_name, model_cls, config_cls)
    train_app.command(name=model_name, help=f"Train {model_name} model")(train_command)

# Fallback command to list available models
@train_app.callback(invoke_without_command=True)
def train_main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo("Available models:")
        for model_name in MODEL_REGISTRY.keys():
            typer.echo(f"  {model_name}")
        typer.echo("\nUse 'aurora train <model_name> --help' for model-specific options.")



# Create subcommand for plotting
plot_app = typer.Typer(help="Plotting utilities")
app.add_typer(plot_app, name="plot")

@plot_app.command("m-emis")
def plot_physical_model_matrix(physical_model_path: Path):
    pm, _ = load_physical_model(physical_model_path, torch.device("cpu"))
    plot_model_matrix(pm)
    
