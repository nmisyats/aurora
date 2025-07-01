"""
Generic plotting utilities for Aurora library.

This module provides reusable plotting functions that can be used both
by the CLI and by users for custom visualization needs.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import pyvista as pv
import numpy as np

from aurora.utils import bounds2d_to_tuple, bounds3d_to_tuple
from aurora.camera import Camera
import aurora.physics as phy


def plot_training_losses(losses: list[float] | dict[str, list[float]]):
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
    energy_edges: Optional[torch.Tensor] = None,
    title: str = "$Q_0$",
    xlabel: str = "y (km)",
    ylabel: str = "x (km)",
    colorbar_label: str = "mW/m²",
    cmap: str = "jet",
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    aspect: str = "equal"
) -> Tuple[Figure, Axes]:
    """
    Plot 2D flux data.
    
    Args:
        flux_data: 2D or 3D tensor. If 3D, will compute total energy flux.
        xy_bounds: Tuple of (xy_min, xy_max) tensors defining spatial bounds
        energy_edges: Energy bin edges for total flux calculation (required if flux_data is 3D)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label  
        colorbar_label: Colorbar label
        cmap: Colormap name
        figsize: Figure size (width, height)
        ax: Existing axes to plot on (optional)
        vmin, vmax: Color scale limits
        show_colorbar: Whether to show colorbar
        aspect: Aspect ratio setting
        
    Returns:
        Tuple of (figure, axes)
    """
    # Process flux data
    if flux_data.dim() == 3:
        if energy_edges is None:
            raise ValueError("energy_edges required for 3D flux data")
        plot_data = phy.compute_total_energy_flux(flux_data, energy_edges).cpu()
    else:
        plot_data = flux_data.cpu()
    
    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get spatial bounds
    xy_min, xy_max = xy_bounds
    x_min, x_max, y_min, y_max = bounds2d_to_tuple(xy_min, xy_max)
    
    # Plot image
    im = ax.imshow(
        plot_data,
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
        cbar.set_label(colorbar_label)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return fig, ax


def plot_flux_1d(
    flux_data: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    energy_edges: torch.Tensor,
    title: str = "Electron flux",
    labels: tuple[str, ...] | None = None,
    xlabel: str = "E [eV]",
    ylabel: str = "f [cm$^{-2}$s$^{-1}$eV$^{-1}$]",
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot 1D flux data.
    
    Args:
        TODO
        
    Returns:
        Tuple of (figure, axes)
    """ 
    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if not isinstance(flux_data, (list, tuple)):
        flux_data = (flux_data,)
    energies = (energy_edges[1:] + energy_edges[:-1]) / 2.0
    
    if labels is not None:
        for data, label in zip(flux_data, labels):
            ax.plot(energies, data, label=label)
        ax.legend()
    else:
        for data in flux_data:
            ax.plot(energies, data)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    return fig, ax


def plot_flux_2d_comparison(
    estimated_flux: torch.Tensor,
    reference_flux: torch.Tensor,
    estimated_bounds: Tuple[torch.Tensor, torch.Tensor],
    reference_bounds: Tuple[torch.Tensor, torch.Tensor],
    energy_edges: Optional[torch.Tensor] = None,
    titles: Tuple[str, str] = ("Reference $Q_0$", "Reconstructed $Q_0$"),
    colorbar_label: str = "mW/m²",
    cmap: str = "jet",
    figsize: Tuple[float, float] = (8, 4),
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
        energy_edges: Energy bin edges (required if flux tensors are 3D)
        titles: Titles for (reference, estimated) plots
        colorbar_label: Colorbar label
        cmap: Colormap name
        figsize: Figure size
        show_mae: Whether to compute and display MAE
        highlight_reconstruction_area: Whether to highlight reconstruction area on reference
        
    Returns:
        Tuple of (figure, list of axes)
    """
    # Process flux data
    if estimated_flux.dim() == 3:
        if energy_edges is None:
            raise ValueError("energy_edges required for 3D flux data")
        est_q0 = phy.compute_total_energy_flux(estimated_flux, energy_edges).cpu()
        ref_q0 = phy.compute_total_energy_flux(reference_flux, energy_edges).cpu()
    else:
        est_q0 = estimated_flux.cpu()
        ref_q0 = reference_flux.cpu()
    
    # Setup figure with shared colorbar
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(1, 2),
        axes_pad=0.1,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad="10%"
    )
    
    # Get bounds
    est_xy_min, est_xy_max = estimated_bounds
    ref_xy_min, ref_xy_max = reference_bounds
    
    x_est_min, x_est_max, y_est_min, y_est_max = bounds2d_to_tuple(est_xy_min, est_xy_max)
    x_ref_min, x_ref_max, y_ref_min, y_ref_max = bounds2d_to_tuple(ref_xy_min, ref_xy_max)
    
    # Determine color range
    vmin = min(est_q0.min(), ref_q0.min())
    vmax = max(est_q0.max(), ref_q0.max())
    
    # Plot reference flux
    grid[0].imshow(
        ref_q0,
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
        est_q0,
        interpolation='none',
        extent=[y_est_min, y_est_max, x_est_max, x_est_min],
        cmap=cmap,
        vmin=vmin, vmax=vmax
    )
    
    # Show MAE if requested
    if show_mae:
        # Extract corresponding region from reference for MAE calculation
        if est_q0.shape != ref_q0.shape:
            # Calculate pixel indices for the estimated region within the reference
            ref_height, ref_width = ref_q0.shape
            
            # Calculate the pixel coordinates of the estimated bounds within the reference grid
            y_start_idx = int((y_est_min - y_ref_min) / (y_ref_max - y_ref_min) * ref_width)
            y_end_idx = int((y_est_max - y_ref_min) / (y_ref_max - y_ref_min) * ref_width)
            x_start_idx = int((x_est_min - x_ref_min) / (x_ref_max - x_ref_min) * ref_height)
            x_end_idx = int((x_est_max - x_ref_min) / (x_ref_max - x_ref_min) * ref_height)
            
            # Extract the corresponding subregion from reference
            ref_q0_cropped = ref_q0[x_start_idx:x_end_idx, y_start_idx:y_end_idx]
            
            # Resize to match estimated tensor if needed
            if ref_q0_cropped.shape != est_q0.shape:
                ref_q0_cropped = torch.nn.functional.interpolate(
                    ref_q0_cropped.unsqueeze(0).unsqueeze(0), 
                    size=est_q0.shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
        else:
            ref_q0_cropped = ref_q0
        mae = torch.mean(torch.abs(est_q0 - ref_q0_cropped))
        grid[1].text(
            0.99, 0.01, f"MAE = {mae:.3f} {colorbar_label.split('/')[0]}",
            transform=grid[1].transAxes,
            ha='right', va='bottom',
            color='white', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3')
        )
    
    # Add colorbar and labels
    cbar = grid[0].cax.colorbar(im)
    cbar.set_label(colorbar_label)
    
    grid[0].set_title(titles[0])
    grid[1].set_title(titles[1])
    grid[0].set_xlabel("y (km)")
    grid[1].set_xlabel("y (km)")
    grid[0].set_ylabel("x (km)")
    
    return fig, [grid[0], grid[1]]


def plot_cameras_grid(
    cameras: List[Camera],
    selected_camera: Optional[str] = None,
    figsize_per_image: float = 3.0,
    colorbar_label: str = "Rayleigh",
    cmap: str = "viridis",
    title_format: str = "{name}",
    global_color_scale: bool = True,
    max_cols: Optional[int] = None
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """
    Plot camera images in a grid layout or single camera.
    
    Args:
        cameras: List of camera objects with .image, .name attributes
        selected_camera: Name of specific camera to plot (plots single image)
        figsize_per_image: Size factor per image
        colorbar_label: Label for colorbar
        cmap: Colormap name
        title_format: Format string for titles (can use camera attributes)
        global_color_scale: Whether to use consistent color scale across all images
        max_cols: Maximum number of columns (auto-computed if None)
        
    Returns:
        Tuple of (figure, axes). Axes is single Axes for one camera, list for multiple.
    """
    if selected_camera is not None:
        # Plot single camera
        cam_dict = {cam.name: cam for cam in cameras}
        if selected_camera not in cam_dict:
            raise ValueError(f"Camera '{selected_camera}' not found")
        
        cam = cam_dict[selected_camera]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cam.image, cmap=cmap)
        cbar = fig.colorbar(im)
        cbar.set_label(colorbar_label)
        
        # Format title with camera attributes
        title = title_format.format(**{
            'name': cam.name,
            'latitude': getattr(cam, 'latitude', 0),
            'longitude': getattr(cam, 'longitude', 0),
            'altitude': getattr(cam, 'altitude', 0)
        })
        ax.set_title(title)
        
        return fig, ax
    
    else:
        # Plot multiple cameras in grid
        n_imgs = len(cameras)
        if n_imgs == 0:
            raise ValueError("No cameras provided")
        
        # Compute grid layout
        if max_cols is not None:
            cols = min(max_cols, n_imgs)
        else:
            cols = math.ceil(math.sqrt(n_imgs))
        rows = math.ceil(n_imgs / cols)
        
        # Determine color scale
        if global_color_scale:
            vmin = min(cam.image.min() for cam in cameras)
            vmax = max(cam.image.max() for cam in cameras)
        else:
            vmin = vmax = None
        
        # Create figure and image grid
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
        
        # Plot images
        im = None
        for ax, cam in zip(grid, cameras):
            im = ax.imshow(cam.image, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            
            # Format title
            title = title_format.format(**{
                'name': cam.name,
                'latitude': getattr(cam, 'latitude', 0),
                'longitude': getattr(cam, 'longitude', 0),
                'altitude': getattr(cam, 'altitude', 0)
            })
            ax.set_title(title)
        
        # Turn off unused axes
        for ax in grid[n_imgs:]:
            ax.axis('off')
        
        # Add shared colorbar
        if im is not None:
            cbar = grid.cbar_axes[0].colorbar(im)
            cbar.set_label(colorbar_label)
        
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
        shade=False
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
    colorbar_label: str = "Rayleigh",
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