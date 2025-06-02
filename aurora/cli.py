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

from aurora.models import MODEL_REGISTRY
from aurora.reconstruction import Reconstruction, Dataset
from aurora.reconstruction import load_reconstruction, save_reonstruction
from aurora.data import (
    load_physical_model,
    load_cameras,
    load_reference_flux,
    load_camera_positions,
    parse_camera
)
from aurora.utils import (
    xy_grid,
    xyz_grid,
    bounds2d_to_tuple,
    bounds3d_to_tuple,
    save_matrix_data,
    save_2d_grid_data,
    save_3d_grid_data
)

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
        ref_flux: Path = typer.Option(None, help="Reference flux to compare the reconstruction with"),
        **config_kwargs
    ):
        typer.echo(f"Training model: {model_name}")
        
        # Choose device
        device = choose_best_device(gpu)
        
        # Load physics model
        pm = load_physical_model(physical_model_path, device)
        
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
        dataset = Dataset(cams, pm.frame, device)
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

            if ref_flux is not None:
                ref = load_reference_flux(ref_flux, device)

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
def plot_flux(
    flux_path: Path = typer.Argument(..., help="Path to flux description"),
    model_path: Path = typer.Argument(..., help="Path to model description")
):
    ref = load_reference_flux(flux_path, "cpu")
    pm = load_physical_model(model_path, "cpu")
    
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

@plot_app.command("cams")
def plot_cameras(
    dataset_path: Path = typer.Argument(..., help="Path to dataset description"),
    cam_name: str = typer.Option(None, help="Name of the camera to plot")
):
    cams = load_cameras(dataset_path)

    if cam_name is not None:
        cams = {cam.name: cam for cam in cams}
        cam = cams[cam_name]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cam.image)
        cbar = ax.figure.colorbar(im)
        cbar.set_label("Rayleigh")
        ax.set_title(f"{cam_name} ({cam.latitude:.3f}°N {cam.longitude:.3f}°E +{cam.altitude:.3f}km)")
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
    reconstruction_path: Path = typer.Argument(..., help="Path to reconstruction"),
    res_x: int = typer.Option(128),
    res_y: int = typer.Option(128),
    gpu: bool = typer.Option(True),
    plot: bool = typer.Option(True),
    save: Path = typer.Option(None)
):
    device = choose_best_device(gpu)
    recon = load_reconstruction(reconstruction_path, device)
    recon.eval_mode()
    pm = recon.physical_model
    xy_min = pm.frame.xy_min
    xy_max = pm.frame.xy_max
    xy = xy_grid(xy_min, xy_max, res_x, res_y)
    f = recon.f(xy).detach()

    if save is not None:
        save_3d_grid_data(f, save)
    
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
    reconstruction_or_reference_path: Path = typer.Argument(..., help="Path to reconstruction or reference flux"),
    physical_model: Path = typer.Option(None, help="Path to physical model (for reference flux only)"),
    res_x: int = typer.Option(100),
    res_y: int = typer.Option(100),
    res_z: int = typer.Option(50),
    gpu: bool = typer.Option(True),
    plot: bool = typer.Option(True),
    save: Path = typer.Option(None)
):
    device = choose_best_device(gpu)
    if reconstruction_or_reference_path.suffix == ".pth":
        recon = load_reconstruction(reconstruction_or_reference_path, device)
        recon.eval_mode()
        pm = recon.physical_model
        xyz_min = pm.frame.box_min
        xyz_max = pm.frame.box_max
        xyz = xyz_grid(xyz_min, xyz_max, res_x, res_y, res_z)
        l = recon.L(xyz).detach().cpu()
    else:
        ref = load_reference_flux(reconstruction_or_reference_path, device)
        if physical_model is None:
            typer.echo("Generating emission from flux data requires a physical model.")
            raise typer.Exit(1)
        pm = load_physical_model(physical_model, device)
        z_min = pm.frame.box_min[2]
        z_max = pm.frame.box_max[2]
        xyz_min = torch.tensor([*ref.xy_min, z_min]).to(device)
        xyz_max = torch.tensor([*ref.xy_max, z_max]).to(device)
        xyz = xyz_grid(xyz_min, xyz_max, res_x, res_y, res_z)
        xy = xyz[...,:2]
        f = ref.f_at(xy)
        l = pm.L(xyz, f).cpu()
    
    if save is not None:
        save_3d_grid_data(l, save)
    
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

@gen_app.command("image")
def generate_images(
    reconstruction_or_reference_path: Path = typer.Argument(..., help="Path to reconstruction"),
    dataset_path: Path = typer.Argument(..., help="Datset description"),
    physical_model: Path = typer.Option(None, help="Path to physical model (for reference flux only)"),
    locations: str = typer.Option(None, parser=lambda s: s.split(",")),
    ray_bins: int = typer.Option(100),
    downsample: int = typer.Option(None),
    gpu: bool = typer.Option(True),
    plot: bool = typer.Option(True),
):
    device = choose_best_device(gpu)
    if reconstruction_or_reference_path.suffix == ".pth":
        device = choose_best_device(gpu)
        recon = load_reconstruction(reconstruction_or_reference_path, device)
    else:
        ref = load_reference_flux(reconstruction_or_reference_path, device)
        if physical_model is None:
            typer.echo("Generating image from flux data requires a physical model.")
            raise typer.Exit(1)
        pm = load_physical_model(physical_model, device)
        recon = Reconstruction(pm, ref, device)
    recon.eval_mode()

    cams = load_cameras(dataset_path)
    if locations is not None:
        cams = list(filter(lambda c: c.name in locations, cams))
    n_cam = len(cams)

    if downsample is not None:
        for i in range(n_cam):
            cams[i] = cams[i].downsample(downsample)

    imgs = []
    for i, cam in enumerate(cams):
        print(f"Generating image {i+1}/{n_cam} ({cam.name})")
        imgs.append(recon.image(cam, ray_bins))

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