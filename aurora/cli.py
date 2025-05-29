import typer
from dataclasses import fields
import torch
import inspect
from pathlib import Path

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import matplotlib.patches as patches
import pyvista as pv

from aurora.models import MODEL_REGISTRY
from aurora.reconstruction import Reconstruction, RayDataset, load_reconstruction, save_reonstruction
from aurora.data import load_physical_model, load_cameras
from aurora.utils import xy_grid, xyz_grid, bounds2d_to_tuple, bounds3d_to_tuple

app = typer.Typer()

def choose_best_device(allow_gpu: bool = True):
    if allow_gpu:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")

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
        save: Path = typer.Option(None, help="Path to file where to save the reconstruction"),
        plot: bool = typer.Option(True, help="Plot the reconstruction after training complete"),
        res_x: int = typer.Option(128, help="x resolution for plotting"),
        res_y: int = typer.Option(128, help="y resolution for plotting"),
        **config_kwargs
    ):
        typer.echo(f"Training model: {model_name}")
        
        # Choose device
        device = choose_best_device(gpu)
        
        # Load physics model
        pm, ref = load_physical_model(physical_model_path, device)
        
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

        if save is not None:
            save_reonstruction(recon, save)

        if plot:
            recon.eval_mode()
            recon_xy = xy_grid(pm.frame.xy_min, pm.frame.xy_max, res_x, res_y)
            estimated_f = recon.f(recon_xy).detach()
            estimated_q0 = pm.q0(estimated_f).cpu()

            x_rec_min, y_rec_min = pm.frame.xy_min.cpu()
            x_rec_max, y_rec_max = pm.frame.xy_max.cpu()

            if ref is not None:
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
                reference_q0 = pm.q0(ref.image).cpu()

                # === Determine color range ===
                vmin = min(estimated_q0.min(), reference_q0.min())
                vmax = max(estimated_q0.max(), reference_q0.max())

                # === Plot reference flux ===
                x_ref_min, y_ref_min = ref.xy_min.cpu()
                x_ref_max, y_ref_max = ref.xy_max.cpu()

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
                ref_f_at_xy = ref.f_at(recon_xy)
                true_q0_at_grid = pm.q0(ref_f_at_xy).cpu()
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
def plot_physical_model_matrix(path: Path = typer.Argument(..., help="Path to physical model")):
    pm, _ = load_physical_model(path, "cpu")
    m = pm.m_mat
    z_edges = pm.z_edges
    E_edges = pm.E_edges

    plt.figure(figsize=(8, 6))

    # Plot with pcolormesh using log scale for m
    log_m = torch.log10(m)
    log_m = torch.nan_to_num(log_m, neginf=torch.min(log_m[log_m != -torch.inf]))
    mesh = plt.pcolormesh(E_edges, z_edges, log_m, shading='auto', cmap='jet')
    
    # Set x to log scale
    plt.xscale('log')

    # Axis labels and colorbar
    plt.xlabel(r"$E$ [eV]")
    plt.ylabel(r"$z$ [km]")
    plt.title(r"$\mathbf{M}$")
    plt.colorbar(mesh, label=r"$\log_{10}(m)$")

    plt.tight_layout()
    plt.show()

@plot_app.command("flux")
def plot_physical_model_flux(
    path: Path = typer.Argument(..., help="Path to physical model"),
):
    pm, ref = load_physical_model(path, "cpu")
    if ref is None:
        typer.echo("The specified model doesn't have a reference flux.", err=True)
        typer.Exit(1)
    
    xy_min = ref.xy_min
    xy_max = ref.xy_max
    q0 = pm.q0(ref.image)
    
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



# Create subcommand for generating
gen_app = typer.Typer(help="Generation utilities")
app.add_typer(gen_app, name="gen")

@gen_app.command("flux")
def generate_reconstructed_flux(
    path: Path = typer.Argument(..., help="Path to reconstruction"),
    res_x: int = typer.Option(128),
    res_y: int = typer.Option(128),
    gpu: bool = typer.Option(True),
    plot: bool = typer.Option(True)
):
    device = choose_best_device(gpu)
    recon = load_reconstruction(path, device)
    recon.eval_mode()
    pm = recon.physical_model
    xy_min = pm.frame.xy_min
    xy_max = pm.frame.xy_max
    xy = xy_grid(xy_min, xy_max, res_x, res_y)
    f = recon.f(xy).detach()
    
    if plot:
        q0 = pm.q0(f)
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
    path: Path = typer.Argument(..., help="Path to reconstruction or physical model"),
    res_x: int = typer.Option(100),
    res_y: int = typer.Option(100),
    res_z: int = typer.Option(50),
    gpu: bool = typer.Option(True),
    plot: bool = typer.Option(True)
):
    device = choose_best_device(gpu)
    if path.suffix == ".pth":
        recon = load_reconstruction(path, device)
        recon.eval_mode()
        pm = recon.physical_model
        xyz_min = pm.frame.box_min
        xyz_max = pm.frame.box_max
        xyz = xyz_grid(xyz_min, xyz_max, res_x, res_y, res_z)
        l = recon.L(xyz).detach().cpu()
    else:
        pm, ref = load_physical_model(path, device)
        if ref is None:
            typer.echo("The specified physical model does not have a reference flux.")
            typer.Exit(1)
        z_min = pm.frame.box_min[2]
        z_max = pm.frame.box_max[2]
        xyz_min = torch.tensor([*ref.xy_min, z_min]).to(device)
        xyz_max = torch.tensor([*ref.xy_max, z_max]).to(device)
        xyz = xyz_grid(xyz_min, xyz_max, res_x, res_y, res_z)
        xy, z = xyz[...,:2], xyz[...,2]
        f = ref.f_at(xy)
        l = pm.L(z, f).cpu()
    
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