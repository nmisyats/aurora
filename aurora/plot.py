"""
Generic plotting utilities for Aurora library.

This module provides reusable plotting functions that can be used both
by the CLI and by users for custom visualization needs.
"""

import math
from typing import List, Optional, Tuple, Dict, Union, Iterable

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation as mpl_animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import pyvista as pv
import numpy as np

from aurora.utils import bounds2d_to_tuple, bounds3d_to_tuple
from aurora.camera import Camera
import aurora.physics as phy


def plot_training_losses(losses: Union[List[float], Dict[str, List[float]]]):
    """
    Plots the training loss over iterations.

    Args:
        losses: The (labelled) loss history to plot.
    
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots()
    if isinstance(losses, dict):
        for name, loss in losses.items():
            ax.plot(loss, label=name)
    else:
        ax.plot(loss)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.set_yscale("log")
    return fig, ax


def plot_flux_2d(
    flux_data: torch.Tensor,
    xy_bounds: Tuple[torch.Tensor, torch.Tensor],
    title: str = "$Q_0$",
    xlabel: str = "y (km)",
    ylabel: str = "x (km)",
    unit: str = "mW/m²",
    cmap: str = "jet",
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    aspect: str = "equal"
) -> Tuple[Figure, Axes]:
    """
    Plot 2D flux scalar data.
    
    Args:
        flux_data: 2D tensor.
        xy_bounds: Tuple of (xy_min, xy_max) tensors defining spatial bounds
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label  
        cmap: Colormap name
        figsize: Figure size (width, height)
        ax: Existing axes to plot on (optional)
        vmin, vmax: Color scale limits
        show_colorbar: Whether to show colorbar
        aspect: Aspect ratio setting
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get spatial bounds
    xy_min, xy_max = xy_bounds
    xy_min = xy_min.cpu()
    xy_max = xy_max.cpu()
    x_min, x_max, y_min, y_max = bounds2d_to_tuple(xy_min, xy_max)
    
    # Plot image
    im = ax.imshow(
        flux_data.cpu(),
        interpolation='none',
        extent=[y_min, y_max, x_max, x_min],
        cmap=cmap,
        aspect=aspect,
        vmin=vmin,
        vmax=vmax
    )
    
    # Add colorbar if requested
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(unit)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return fig, ax


def plot_flux_1d(
    flux_data: Union[torch.Tensor, Iterable[torch.Tensor]],
    energy_edges: torch.Tensor,
    title: str = "Electron flux",
    labels: Optional[Tuple[str, ...]] = None,
    xlabel: str = "E [eV]",
    ylabel: str = "f [m$^{-2}$s$^{-1}$eV$^{-1}$]",
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot 1D flux data.
    
    Args:
        flux_data: 1D tensor or list of 1D tensor, each corresponding to a flux spectrum to plot
        energy_edges: 1D tensor defining the energy bins of the flux
        title: Plot title
        labels: Label for each plot
        xlabel: X-axis label
        ylabel: Y-axis label  
        figsize: Figure size (width, height)
        ax: Existing axes to plot on (optional)
        
    Returns:
        Tuple of (figure, axes)
    """ 
    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if isinstance(flux_data, torch.Tensor):
        flux_data = (flux_data.cpu(),)
    else:
        flux_data = tuple(f.cpu() for f in flux_data)
    energy_edges = energy_edges.cpu()
    energies = (energy_edges[1:] + energy_edges[:-1]) / 2.0
    
    scale = 10**3 * torch.pi
    if labels is not None:
        for data, label in zip(flux_data, labels):
            ax.plot(energies, scale * data, label=label, marker="x")
        ax.legend()
    else:
        for data in flux_data:
            ax.plot(energies, scale * data, marker="x")
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    return fig, ax


def plot_flux_2d_comparison(
    est_data: torch.Tensor,
    ref_data: torch.Tensor,
    est_bounds: Tuple[torch.Tensor, torch.Tensor],
    ref_bounds: Tuple[torch.Tensor, torch.Tensor],
    titles: Tuple[str, str] = ("Reference $Q_0$", "Reconstructed $Q_0$"),
    xlabel: str = "y (km)",
    ylabel: str = "x (km)",
    unit: str = "mW/m²",
    cmap: str = "jet",
    figsize: Tuple[float, float] = (12, 6),
    show_mae: bool = True,
    highlight_reconstruction_area: bool = True
) -> Tuple[Figure, List[Axes]]:
    """
    Plot side-by-side comparison of estimated vs reference flux.
    
    Args:
        estimated_flux: Estimated flux tensor
        reference_flux: Reference flux tensor  
        estimated_bounds: Spatial bounds for estimated flux
        reference_bounds: Spatial bounds for reference flux
        titles: Titles for (reference, estimated) plots
        cmap: Colormap name
        figsize: Figure size
        show_mae: Whether to compute and display MAE
        highlight_reconstruction_area: Whether to highlight reconstruction area on reference
        
    Returns:
        Tuple of (figure, list of axes)
    """
    # Process flux data
    est_data = est_data.cpu()
    ref_data = ref_data.cpu()
    
    fig, grid = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=False)
    
    # Get bounds
    est_xy_min, est_xy_max = est_bounds
    ref_xy_min, ref_xy_max = ref_bounds
    est_xy_min = est_xy_min.cpu()
    est_xy_max = est_xy_max.cpu()
    ref_xy_min = ref_xy_min.cpu()
    ref_xy_max = ref_xy_max.cpu()
    
    x_est_min, x_est_max, y_est_min, y_est_max = bounds2d_to_tuple(est_xy_min, est_xy_max)
    x_ref_min, x_ref_max, y_ref_min, y_ref_max = bounds2d_to_tuple(ref_xy_min, ref_xy_max)
    
    # Determine color range from finite values only (keep NaNs in image data for masking effects)
    finite_est = est_data[torch.isfinite(est_data)]
    finite_ref = ref_data[torch.isfinite(ref_data)]
    mins = []
    maxs = []
    if finite_est.numel() > 0:
        mins.append(finite_est.min().item())
        maxs.append(finite_est.max().item())
    if finite_ref.numel() > 0:
        mins.append(finite_ref.min().item())
        maxs.append(finite_ref.max().item())
    if mins and maxs:
        vmin = min(mins)
        vmax = max(maxs)
    else:
        vmin = vmax = None
    
    # Plot reference flux
    grid[0].imshow(
        ref_data,
        interpolation='none',
        extent=[y_ref_min, y_ref_max, x_ref_max, x_ref_min],
        cmap=cmap,
        vmin=vmin, vmax=vmax
    )
    
    # Highlight reconstruction area if requested
    if highlight_reconstruction_area:
        rect = patches.Rectangle(
            (y_est_min, x_est_min),
            y_est_max - y_est_min,
            x_est_max - x_est_min,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        grid[0].add_patch(rect)
    
    # Plot estimated flux
    im = grid[1].imshow(
        est_data,
        interpolation='none',
        extent=[y_est_min, y_est_max, x_est_max, x_est_min],
        cmap=cmap,
        vmin=vmin, vmax=vmax
    )

    x_min = min(x_ref_min, x_est_min)
    x_max = max(x_ref_max, x_est_max)
    for ax in grid:
        ax.set_ylim(x_max, x_min)  # keep your “origin at top” convention
    
    # Show MAE if requested
    if show_mae:
        # Extract corresponding region from reference for MAE calculation
        if est_data.shape != ref_data.shape:
            # Calculate pixel indices for the estimated region within the reference
            ref_height, ref_width = ref_data.shape
            
            # Calculate the pixel coordinates of the estimated bounds within the reference grid
            y_start_idx = int((y_est_min - y_ref_min) / (y_ref_max - y_ref_min) * ref_width)
            y_end_idx = int((y_est_max - y_ref_min) / (y_ref_max - y_ref_min) * ref_width)
            x_start_idx = int((x_est_min - x_ref_min) / (x_ref_max - x_ref_min) * ref_height)
            x_end_idx = int((x_est_max - x_ref_min) / (x_ref_max - x_ref_min) * ref_height)
            
            # Extract the corresponding subregion from reference
            ref_est_cropped = ref_data[x_start_idx:x_end_idx, y_start_idx:y_end_idx]
            
            # Resize to match estimated tensor if needed
            if ref_est_cropped.shape != est_data.shape:
                ref_est_cropped = torch.nn.functional.interpolate(
                    ref_est_cropped.unsqueeze(0).unsqueeze(0), 
                    size=est_data.shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
        else:
            ref_est_cropped = ref_data
        abs_err = torch.abs(est_data - ref_est_cropped)
        finite_err = abs_err[torch.isfinite(abs_err)]
        if finite_err.numel() > 0:
            mae_text = f"MAE = {finite_err.mean().item():.3f} {unit}"
        else:
            mae_text = f"MAE = n/a {unit}"
        grid[1].text(
            0.99, 0.01, mae_text,
            transform=grid[1].transAxes,
            ha='right', va='bottom',
            color='white', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3')
        )
    
    # Add colorbar and labels
    cbar = fig.colorbar(im, ax=grid, fraction=0.046, pad=0.04)
    cbar.set_label(unit)
    
    grid[0].set_title(titles[0])
    grid[1].set_title(titles[1])
    grid[0].set_xlabel(xlabel)
    grid[1].set_xlabel(xlabel)
    grid[0].set_ylabel(ylabel)
    
    return fig, [grid[0], grid[1]]


def plot_model_matrices(
    matrices: Union[torch.Tensor, Iterable[torch.Tensor]],
    altitude_bins: torch.Tensor,
    energy_bins: torch.Tensor,
    titles: Optional[Tuple[str, ...]] = None,
    xlabel: str = "Energy range [eV]",
    ylabel: str = "Altitude range [km]",
    title: str = "Model matrix",
    unit: str = "a.u.",
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (10, 5),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    max_bin_labels: int = 8,
    aspect: str = "auto"
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """
    Plot one or more emission/density matrices as 2D heatmaps.
    Matrix values are transformed with ``log10`` before plotting.

    Expected matrix layout is ``(n_altitude_bins, n_energy_bins)``. If a matrix
    is provided as ``(n_energy_bins, n_altitude_bins)``, it is transposed
    automatically.

    Args:
        matrices: A 2D tensor, a stack of 2D tensors, or an iterable of 2D
            tensors.
        altitude_bins: Altitude bin edges of shape ``(n_altitude_bins + 1,)``.
        energy_bins: Energy bin edges of shape ``(n_energy_bins + 1,)``.
        titles: Optional per-panel titles.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Base title used when multiple matrices are provided and `titles`
            is not set.
        unit: Unit string used for colorbar label as ``log10(unit)``.
        cmap: Matplotlib colormap name.
        figsize: Figure size.
        vmin: Optional lower color limit (in log10 domain).
        vmax: Optional upper color limit (in log10 domain).
        show_colorbar: Whether to draw a colorbar.
        max_bin_labels: Target number of tick labels per axis.
        aspect: Matplotlib aspect mode.

    Returns:
        ``(fig, ax)`` for a single matrix, or ``(fig, axes)`` for multiple
        matrices.
    """
    # Accept one 2D matrix, one 3D tensor (stack), or iterable of 2D tensors
    if isinstance(matrices, torch.Tensor):
        if matrices.ndim == 2:
            matrices = (matrices.cpu(),)
        else:
            matrices = tuple(m.cpu() for m in matrices)
    else:
        matrices = tuple(m.cpu() for m in matrices)

    altitude_bins = altitude_bins.cpu().flatten()
    energy_bins = energy_bins.cpu().flatten()
    n_alt = altitude_bins.numel() - 1
    n_energy = energy_bins.numel() - 1

    # Aurora matrices are (n_alt, n_energy); if swapped, transpose.
    # Plot log10 values and keep non-finite values as NaN so they are not drawn.
    mats = []
    for mat in matrices:
        if mat.shape == (n_energy, n_alt):
            mat = mat.T
        mat = torch.log10(mat)
        mat = torch.where(torch.isfinite(mat), mat, torch.nan)
        mats.append(mat)

    # Shared color range across all panels (ignore NaN/inf)
    finite_vals = [mat[torch.isfinite(mat)] for mat in mats]
    finite_vals = [vals for vals in finite_vals if vals.numel() > 0]
    if finite_vals:
        all_vals = torch.cat(finite_vals)
        if vmin is None:
            vmin = all_vals.min().item()
        if vmax is None:
            vmax = all_vals.max().item()

    # Ticks as actual values (not ranges), uniformly spaced along each axis
    n_ticks = max(2, max_bin_labels)
    e_min, e_max = energy_bins[0].item(), energy_bins[-1].item()
    z_min, z_max = altitude_bins[0].item(), altitude_bins[-1].item()
    e_ticks = np.geomspace(e_min, e_max, n_ticks)
    z_ticks = np.linspace(z_min, z_max, n_ticks)
    e_labels = [f"{v:.3g}" for v in e_ticks]
    z_labels = [f"{v:.3g}" for v in z_ticks]

    if len(mats) == 1:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(
            energy_bins.numpy(),
            altitude_bins.numpy(),
            mats[0].numpy(),
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(titles[0] if titles is not None else title)
        ax.set_xscale("log")
        ax.set_aspect(aspect)
        ax.set_xlim(e_min, e_max)
        ax.set_ylim(z_min, z_max)
        ax.set_xticks(e_ticks)
        ax.set_yticks(z_ticks)
        ax.set_xticklabels(e_labels, rotation=45, ha="right")
        ax.set_yticklabels(z_labels)
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(f"log10({unit})" if unit else "log10(value)")
        return fig, ax

    n = len(mats)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, grid = plt.subplots(rows, cols, figsize=figsize, squeeze=False, sharey=True)
    axes = grid.flatten()
    im = None

    for i, (ax, mat) in enumerate(zip(axes, mats)):
        im = ax.pcolormesh(
            energy_bins.numpy(),
            altitude_bins.numpy(),
            mat.numpy(),
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        ax.set_xlabel(xlabel)
        if titles is not None:
            ax.set_title(titles[i])
        else:
            ax.set_title(f"{title} {i+1}")
        ax.set_xscale("log")
        ax.set_aspect(aspect)
        ax.set_xlim(e_min, e_max)
        ax.set_ylim(z_min, z_max)
        ax.set_xticks(e_ticks)
        ax.set_yticks(z_ticks)
        ax.set_xticklabels(e_labels, rotation=45, ha="right")
        if i % cols == 0:
            ax.set_ylabel(ylabel)
            ax.set_yticklabels(z_labels)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    for ax in axes[n:]:
        ax.axis("off")

    if show_colorbar and im is not None:
        cbar = fig.colorbar(im, ax=axes[:n].tolist(), fraction=0.046, pad=0.04)
        cbar.set_label(f"log10({unit})" if unit else "log10(value)")

    return fig, list(axes[:n])


def plot_cameras_grid(
    cameras: List[Camera],
    selected_camera: Optional[str] = None,
    image_index: int = 0,
    anim_range: Optional[Tuple[int, int]] = None,
    anim_interval_ms: int = 200,
    anim_repeat: bool = True,
    figsize_per_image: float = 3.0,
    colorbar_label: str = "kR",
    cmap: str = "viridis",
    title_format: str = "{camera_id}",
    global_color_scale: bool = True,
    max_cols: Optional[int] = None
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """
    Plot camera images in a grid layout or single camera.
    
    Args:
        cameras: List of camera objects.
        selected_camera: Camera selector to plot a single image. Accepted values:
            "<location>:<wavelength>", "<location>", or camera folder name.
        image_index: Index of the image in camera.images to load and plot
            (ignored when anim_range is provided).
        anim_range: Inclusive `(start, end)` image-index range to animate.
            For example `(5, 15)` animates image indices 5..15.
        anim_interval_ms: Animation frame interval in milliseconds.
        anim_repeat: Whether animation should loop when finished.
        figsize_per_image: Size factor per image
        colorbar_label: Label for colorbar
        cmap: Colormap name
        title_format: Format string for titles. Available fields are:
            camera_id, name, location, wavelength, latitude, longitude,
            altitude, data_path, image_index, image_name, image_path, date_time.
        global_color_scale: Whether to use consistent color scale across all images
        max_cols: Maximum number of columns (auto-computed if None)
        
    Returns:
        Tuple of (figure, axes). Axes is single Axes for one camera, list for multiple.
    """
    if len(cameras) == 0:
        raise ValueError("No cameras provided")
    if image_index < 0:
        raise ValueError(f"image_index must be >= 0, got {image_index}")
    if anim_interval_ms <= 0:
        raise ValueError(f"anim_interval_ms must be > 0, got {anim_interval_ms}")
    if max_cols is not None and max_cols <= 0:
        raise ValueError(f"max_cols must be > 0, got {max_cols}")
    if anim_range is not None:
        anim_start, anim_end = anim_range
        if anim_start < 0 or anim_end < 0:
            raise ValueError(f"anim_range indices must be >= 0, got {anim_range}")
        if anim_end < anim_start:
            raise ValueError(f"anim_range end must be >= start, got {anim_range}")
        frame_indices = list(range(anim_start, anim_end + 1))
    else:
        frame_indices = [image_index]

    def _camera_id(cam: Camera) -> str:
        base = cam.location if cam.location is not None else cam.data_path.name
        if cam.wavelength is not None:
            return f"{base}:{cam.wavelength}"
        return base

    def _camera_aliases(cam: Camera) -> set[str]:
        aliases = {_camera_id(cam), cam.data_path.name}
        if cam.location is not None:
            aliases.add(cam.location)
        return aliases

    image_cache: Dict[Tuple[int, int], Tuple[np.ndarray, object]] = {}

    def _load_plot_image(cam: Camera, idx: int) -> Tuple[np.ndarray, object]:
        key = (id(cam), idx)
        if key in image_cache:
            return image_cache[key]
        if idx >= len(cam.images):
            raise ValueError(
                f"Camera '{_camera_id(cam)}' has only {len(cam.images)} image(s); "
                f"cannot use image_index={idx}."
            )
        img_ref = cam.images[idx]
        img = img_ref.load_image()
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        else:
            img = np.asarray(img)
        image_cache[key] = (img / 1000.0, img_ref)
        return image_cache[key]

    def _format_title(cam_id: str, cam: Camera, img_ref: object, idx: int) -> str:
        date_time = ""
        image_name = ""
        image_path = ""
        if hasattr(img_ref, "date_time") and img_ref.date_time is not None:
            date_time = img_ref.date_time.isoformat(sep=" ")
        if hasattr(img_ref, "path"):
            image_name = img_ref.path.name
            image_path = str(img_ref.path)
        fields = {
            "camera_id": cam_id,
            "name": cam_id,
            "location": cam.location if cam.location is not None else cam.data_path.name,
            "wavelength": cam.wavelength if cam.wavelength is not None else "",
            "latitude": cam.latitude,
            "longitude": cam.longitude,
            "altitude": cam.altitude,
            "data_path": str(cam.data_path),
            "image_index": idx,
            "image_name": image_name,
            "image_path": image_path,
            "date_time": date_time,
        }
        try:
            return title_format.format(**fields)
        except KeyError as exc:
            bad_field = exc.args[0]
            available = ", ".join(sorted(fields.keys()))
            raise ValueError(
                f"Unknown title field '{bad_field}'. Available fields: {available}"
            ) from exc

    if selected_camera is not None:
        matches = [cam for cam in cameras if selected_camera in _camera_aliases(cam)]
        if len(matches) == 0:
            available = ", ".join(_camera_id(cam) for cam in cameras)
            raise ValueError(
                f"Camera '{selected_camera}' not found. Available cameras: {available}"
            )
        if len(matches) > 1:
            available = ", ".join(_camera_id(cam) for cam in matches)
            raise ValueError(
                f"Camera selector '{selected_camera}' is ambiguous. Matches: {available}. "
                "Use '<location>:<wavelength>'."
            )
        cameras_to_plot = [matches[0]]
    else:
        cameras_to_plot = cameras

    first_idx = frame_indices[0]
    plotted = []
    for cam in cameras_to_plot:
        img, img_ref = _load_plot_image(cam, first_idx)
        plotted.append((_camera_id(cam), cam, img, img_ref))

    if global_color_scale:
        finite_mins = []
        finite_maxs = []
        for cam in cameras_to_plot:
            for idx in frame_indices:
                img, _ = _load_plot_image(cam, idx)
                finite = img[np.isfinite(img)]
                if finite.size > 0:
                    finite_mins.append(float(finite.min()))
                    finite_maxs.append(float(finite.max()))
        if len(finite_mins) > 0:
            vmin = min(finite_mins)
            vmax = max(finite_maxs)
        else:
            vmin = vmax = None
    else:
        vmin = vmax = None

    if selected_camera is not None:
        cam_id, cam, img, img_ref = plotted[0]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im)
        cbar.set_label(colorbar_label)
        ax.set_title(_format_title(cam_id, cam, img_ref, first_idx))

        if anim_range is not None:
            def _update_single(frame_pos: int):
                frame_idx = frame_indices[frame_pos]
                frame_img, frame_ref = _load_plot_image(cam, frame_idx)
                im.set_data(frame_img)
                if not global_color_scale:
                    finite = frame_img[np.isfinite(frame_img)]
                    if finite.size > 0:
                        im.set_clim(float(finite.min()), float(finite.max()))
                ax.set_title(_format_title(cam_id, cam, frame_ref, frame_idx))
                return (im,)

            fig._aurora_animation = mpl_animation.FuncAnimation(
                fig,
                _update_single,
                frames=len(frame_indices),
                interval=anim_interval_ms,
                repeat=anim_repeat,
                blit=False,
            )
        return fig, ax

    n_imgs = len(plotted)
    if max_cols is not None:
        cols = min(max_cols, n_imgs)
    else:
        cols = math.ceil(math.sqrt(n_imgs))
    rows = math.ceil(n_imgs / cols)

    fig = plt.figure(figsize=(cols * figsize_per_image, rows * figsize_per_image))
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(rows, cols),
        axes_pad=0.4,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.1
    )

    im = None
    artists: List[Tuple[Axes, AxesImage, str, Camera]] = []
    for ax, (cam_id, cam, img, img_ref) in zip(grid, plotted):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        ax.set_title(_format_title(cam_id, cam, img_ref, first_idx))
        artists.append((ax, im, cam_id, cam))

    for ax in grid[n_imgs:]:
        ax.axis("off")

    if im is not None:
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.set_label(colorbar_label)

    if anim_range is not None:
        def _update_grid(frame_pos: int):
            frame_idx = frame_indices[frame_pos]
            updated = []
            for ax, im_artist, cam_id, cam in artists:
                frame_img, frame_ref = _load_plot_image(cam, frame_idx)
                im_artist.set_data(frame_img)
                if not global_color_scale:
                    finite = frame_img[np.isfinite(frame_img)]
                    if finite.size > 0:
                        im_artist.set_clim(float(finite.min()), float(finite.max()))
                ax.set_title(_format_title(cam_id, cam, frame_ref, frame_idx))
                updated.append(im_artist)
            return tuple(updated)

        fig._aurora_animation = mpl_animation.FuncAnimation(
            fig,
            _update_grid,
            frames=len(frame_indices),
            interval=anim_interval_ms,
            repeat=anim_repeat,
            blit=False,
        )

    return fig, list(grid[:n_imgs])


def plot_volume_3d(
    volume_data: torch.Tensor,
    xyz_bounds: Tuple[torch.Tensor, torch.Tensor],
    scalars_name: str = "density",
    cmap: str = "coolwarm",
    opacity: Optional[List[float]] = None,
    window_size: Tuple[int, int] = (800, 600),
    show_axes: bool = True,
    show_bounds: bool = True,
    background_color: str = "white"
) -> pv.Plotter:
    """
    Create 3D volume visualization using PyVista.
    
    Args:
        volume_data: 3D tensor with shape (x, y, z)
        xyz_bounds: Tuple of (xyz_min, xyz_max) tensors
        scalars_name: Name for the scalar field
        cmap: Colormap name
        opacity: Opacity transfer function values
        window_size: Window size (width, height)
        show_axes: Whether to show coordinate axes
        show_bounds: Whether to show bounding box
        background_color: Background color
        
    Returns:
        PyVista plotter object (call .show() to display)
    """
    if opacity is None:
        opacity = [0, 0.1, 0.3, 0.6, 0.8, 1.0, 1.0]
    
    # Convert to numpy and get dimensions
    volume_np = volume_data.cpu().numpy()
    res_x, res_y, res_z = volume_np.shape
    
    # Create PyVista grid
    grid = pv.ImageData()
    grid.dimensions = (res_x + 1, res_y + 1, res_z + 1)
    
    # Set spacing and origin
    xyz_min, xyz_max = xyz_bounds
    xyz_min = xyz_min.cpu()
    xyz_max = xyz_max.cpu()
    x_min, x_max, y_min, y_max, z_min, z_max = bounds3d_to_tuple(xyz_min, xyz_max)
    
    # Scale to make visualization more reasonable
    x_scale = (x_max - x_min) / (z_max - z_min)
    y_scale = (y_max - y_min) / (z_max - z_min)
    z_scale = 1.0
    
    grid.spacing = (x_scale, y_scale, z_scale)
    grid.origin = (0, 0, 0)
    
    # Add data to grid (PyVista expects Fortran order)
    grid.cell_data[scalars_name] = volume_np.flatten(order="F")
    
    # Create plotter
    pl = pv.Plotter(window_size=window_size)
    pl.background_color = background_color
    
    # Add volume
    pl.add_volume(
        grid,
        scalars=scalars_name,
        cmap=cmap,
        opacity=opacity,
        shade=False,
        scalar_bar_args=dict(
            title=scalars_name,
            vertical=True,
            title_font_size=16,
            label_font_size=12,
            fmt="%.2e",
            n_labels=5,
            italic=False,
            width=0.08,
            height=0.6,
            position_x=0.87,
            position_y=0.2,
        ),
    )
    
    # Optional additions
    if show_axes:
        pl.show_axes()
    if show_bounds:
        pl.add_bounding_box()
    
    return pl


def plot_image_comparison(
    generated_images: List[torch.Tensor],
    reference_images: List[torch.Tensor],
    camera_names: List[str],
    figsize_per_col: float = 2.0,
    colorbar_label: str = "kR",
    cmap: str = "viridis",
    global_color_scale: bool = True,
    row_labels: Tuple[str, str] = ("Generated", "Reference")
) -> Tuple[Figure, List[List[Axes]]]:
    """
    Plot comparison between generated and reference images in a 2-row grid.
    
    Args:
        generated_images: List of generated image tensors
        reference_images: List of reference image tensors  
        camera_names: List of camera names for column titles
        figsize_per_col: Figure size factor per column
        colorbar_label: Label for shared colorbar
        cmap: Colormap name
        global_color_scale: Whether to use consistent color scale
        row_labels: Labels for (generated, reference) rows
        
    Returns:
        Tuple of (figure, list of axes rows)
    """
    n_cams = len(generated_images)
    if len(reference_images) != n_cams or len(camera_names) != n_cams:
        raise ValueError("All input lists must have same length")
    
    # Convert to numpy and determine color scale
    gen_imgs = [img.cpu().detach().numpy() for img in generated_images]
    ref_imgs = [img.cpu().detach().numpy() if torch.is_tensor(img) else img 
                for img in reference_images]
    
    # Convert to kR
    gen_imgs = [img / 1000 for img in gen_imgs]
    ref_imgs = [img / 1000 for img in ref_imgs]

    if global_color_scale:
        all_imgs = gen_imgs + ref_imgs
        vmin = min(img.min() for img in all_imgs)
        vmax = max(img.max() for img in all_imgs)
    else:
        vmin = vmax = None
    
    # Create figure with ImageGrid
    fig = plt.figure(figsize=(n_cams * figsize_per_col + 3.0, 4 + 0.5))
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(2, n_cams),
        axes_pad=0.1,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad="10%"
    )
    
    # Plot images
    all_imgs_flat = gen_imgs + ref_imgs
    im = None
    
    for i, (ax, img) in enumerate(zip(grid, all_imgs_flat)):
        im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add column titles only for top row
        if i < n_cams:
            ax.set_title(camera_names[i])
    
    # Add shared colorbar
    if im is not None:
        cbar = grid[0].cax.colorbar(im)
        cbar.set_label(colorbar_label)
    
    # Add row labels
    grid[0].set_ylabel(row_labels[0])
    grid[n_cams].set_ylabel(row_labels[1])
    
    # Organize axes into rows for return
    axes_rows = [list(grid[:n_cams]), list(grid[n_cams:2*n_cams])]
    
    return fig, axes_rows

def plot_3d_scatter(points, values, xlim=None, ylim=None, zlim=None, unit='m$^{-3}$'):
    """
    Create a 3D scatter plot with equal aspect ratio and color-coded values.
    
    Args:
        points: Tensor of shape (n, 3) containing 3D coordinates of points
        values: Tensor of shape (n,) containing scalar values for each point
        xlim: (xmin, xmax) for x-axis range. If None, deduced from data.
        ylim: (ymin, ymax) for y-axis range. If None, deduced from data.
        zlim: (zmin, zmax) for z-axis range. If None, deduced from data.
        unit: Unit label for the colorbar
    
    Returns:
        A tuple (fig, ax) of the matplotlib figure and axis objects
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Deduce ranges from data if not provided
    if xlim is None:
        xlim = (points[:, 0].min(), points[:, 0].max())
    if ylim is None:
        ylim = (points[:, 1].min(), points[:, 1].max())
    if zlim is None:
        zlim = (points[:, 2].min(), points[:, 2].max())
    
    # Create scatter plot with jet colormap
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=values, cmap='jet', s=5)
    
    # Add colorbar with unit
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label(unit, rotation=270, labelpad=20)
    
    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    # Set axis labels
    ax.set_xlabel('$x$ (km)')
    ax.set_ylabel('$y$ (km)')
    ax.set_zlabel('$z$ (km)')
    
    # Calculate ranges for equal aspect ratio
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]
    
    # Set equal aspect ratio
    max_range = max(x_range, y_range, z_range)
    ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])
    
    plt.tight_layout()
    return fig, ax

# Context manager for plot styling
class PlotStyle:
    """Context manager for temporary plot styling."""
    
    def __init__(self, **kwargs):
        self.style_dict = kwargs
        self.old_params = {}
    
    def __enter__(self):
        for key, value in self.style_dict.items():
            if hasattr(plt, key):
                self.old_params[key] = getattr(plt, key)
                setattr(plt, key, value)
            elif key in plt.rcParams:
                self.old_params[key] = plt.rcParams[key]
                plt.rcParams[key] = value
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.old_params.items():
            if hasattr(plt, key):
                setattr(plt, key, value)
            elif key in plt.rcParams:
                plt.rcParams[key] = value
