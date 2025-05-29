from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pyvista as pv
from pathlib import Path
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Callable
import torch

from aurora.camera import Camera
from aurora.reconstruction import PhysicalModel
from aurora.data import ReferenceFlux
from aurora.utils import load_matrix_data, xy_grid, xyz_grid, mean_absolute_error, CoordLike, bounds_to_tuple

def plot_training_loss(loss: list[float]):
    plt.plot(loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training batch loss")
    plt.show()

def plot_camera_image(cam_path: Path):
    plt.imshow(load_matrix_data(cam_path / "image.dat"))
    plt.colorbar()
    plt.title(cam_path.stem)
    plt.show()

def plot_camera_views(cam_path: Path):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), layout='compressed')
    
    im1 = ax1.imshow(load_matrix_data(cam_path / "az_cam.dat"))
    ax1.set_title("azimuth")
    fig.colorbar(im1, orientation='vertical')

    im2 = ax2.imshow(load_matrix_data(cam_path / "ze_cam.dat"))
    ax2.set_title("zenith")
    fig.colorbar(im2, orientation='vertical')

    plt.show()


# def plot_total_energy_flux(
#     estimate_f: Callable[[torch.Tensor], torch.Tensor],
#     pm: PhysicalModel,
#     res_x: int,
#     res_y: int,
#     reference: ReferenceFlux | None = None
# ):
#     # === Generate xy grid and estimate flux ===
#     xy_grid_tensor = xy_grid(pm.frame.xy_min, pm.frame.xy_max, res_x, res_y)
#     estimated_flux = estimate_f(xy_grid_tensor.reshape(res_x * res_y, 2))
#     estimated_q0 = pm.q0(estimated_flux).reshape((res_x, res_y)).detach().cpu()

#     x_rec_min, y_rec_min = pm.frame.xy_min.cpu()
#     x_rec_max, y_rec_max = pm.frame.xy_max.cpu()

#     if reference is not None:
#         # === Setup figure with 2 subplots and shared colorbar ===
#         fig = plt.figure(figsize=(8, 4))
#         grid = ImageGrid(fig, 111,
#                          nrows_ncols=(1, 2),
#                          axes_pad=0.1,
#                          cbar_location="right",
#                          cbar_mode="single",
#                          cbar_size="7%",
#                          cbar_pad="10%")

#         # === Process reference flux ===
#         reference_flux_tensor = reference.image  # shape: (h, w, bins)
#         h, w, n_bins = reference_flux_tensor.shape
#         reference_flux_flat = reference_flux_tensor.reshape(h * w, n_bins)
#         reference_full_q0 = pm.q0(reference_flux_flat).reshape((h, w)).detach().cpu()

#         # === Determine color range ===
#         vmin = min(estimated_q0.min(), reference_full_q0.min())
#         vmax = max(estimated_q0.max(), reference_full_q0.max())

#         # === Plot reference flux ===
#         x_ref_min, y_ref_min = reference.xy_min.cpu()
#         x_ref_max, y_ref_max = reference.xy_max.cpu()

#         grid[0].imshow(reference_full_q0,
#                        interpolation='none',
#                        extent=[y_ref_min, y_ref_max, x_ref_max, x_ref_min],
#                        cmap="jet",
#                        vmin=vmin, vmax=vmax)

#         # Highlight reconstructed area on reference plot
#         rect = patches.Rectangle((y_rec_min, x_rec_min),
#                                  y_rec_max - y_rec_min,
#                                  x_rec_max - x_rec_min,
#                                  linewidth=1, edgecolor='r', facecolor='none')
#         grid[0].add_patch(rect)

#         # === Plot reconstructed flux ===
#         im = grid[1].imshow(estimated_q0,
#                             interpolation='none',
#                             extent=[y_rec_min, y_rec_max, x_rec_max, x_rec_min],
#                             cmap="jet",
#                             vmin=vmin, vmax=vmax)

#         # === Compute and show MAE ===
#         true_flux_at_grid = reference.f_at(xy_grid_tensor.reshape(res_x * res_y, 2))
#         true_flux_at_grid_flat = true_flux_at_grid.reshape((res_x * res_y, n_bins))
#         true_q0_at_grid = pm.q0(true_flux_at_grid_flat).reshape((res_x, res_y)).detach().cpu()
#         mae = mean_absolute_error(estimated_q0, true_q0_at_grid)
#         grid[1].text(0.99, 0.01, f"MAE = {mae:.3f} mW/m$^2$",
#                      transform=grid[1].transAxes,
#                      ha='right', va='bottom',
#                      color='white', fontsize=10,
#                      bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

#         # === Add colorbar and labels ===
#         cbar = grid[0].cax.colorbar(im)
#         cbar.set_label("mW/m$^2$")

#         grid[0].set_title("Reference $Q_0$")
#         grid[1].set_title("Reconstructed $Q_0$")
#         grid[0].set_xlabel("y (km)")
#         grid[1].set_xlabel("y (km)")
#         grid[0].set_ylabel("x (km)")

#     else:
#         # === Plot only the reconstructed flux ===
#         plt.imshow(estimated_q0,
#                    interpolation='none',
#                    extent=[y_rec_min, y_rec_max, x_rec_max, x_rec_min],
#                    cmap="jet")
#         cbar = plt.colorbar()
#         cbar.set_label("mW/m$^2$")
#         plt.xlabel("y (km)")
#         plt.ylabel("x (km)")
#         plt.title("Reconstructed total energy flux")

#     plt.show()

def plot_reconstructed_images(render_image: Callable[[Camera], torch.Tensor], cams: list[Camera]):
    n_img = len(cams)
    # fig, axs = plt.subplots(2, n_img, figsize=(n_img * 2, 4 + 0.5))  # Added extra space for colorbar
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig = plt.figure(figsize=(n_img * 2, 4 + 0.5))

    # Lists to store min and max values for color scaling
    all_mins = []
    all_maxs = []

    imgs = []
    refs = []

    # First pass to get min and max values across all images
    for i in range(n_img):
        print(f"Generating image {i+1}/{n_img}")
        img = render_image(cams[i]).detach().cpu().numpy()
        ref = cams[i].image.cpu().numpy()
        imgs.append(img)
        refs.append(ref)
        
        all_mins.append(min(img.min(), ref.min()))
        all_maxs.append(max(img.max(), ref.max()))

    # Get global min and max
    vmin = min(all_mins)
    vmax = max(all_maxs)

    # Second pass to plot images with consistent color scale
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, n_img),
                     axes_pad=0.1,
                     cbar_location="right", cbar_mode="single", cbar_size="7%", cbar_pad="10%")
    
    cat = [*imgs, *refs]
    ims = []
    for i, (ax, img) in enumerate(zip(grid, cat)):
        im = ax.imshow(img, vmin=vmin, vmax=vmax)
        ims.append(im)
        ax.set_xticks([])
        ax.set_yticks([])
        if i < n_img:
            ax.set_title(cams[i].name)
    
    cbar = grid[0].cax.colorbar(ims[0])
    cbar.set_label("Rayleigh")

    grid[0].set_ylabel("Generated image")
    grid[n_img].set_ylabel("Reference image")

    plt.show()
    return imgs

def plot_volume_emission(estimate_L: Callable[[torch.Tensor], torch.Tensor], pm: PhysicalModel, res_x: int, res_y: int, res_z: int):
    xyz = xyz_grid(pm.frame.box_min, pm.frame.box_max, res_x, res_y, res_z)
    L = estimate_L(xyz.reshape(res_x*res_y*res_z, 3))
    L = L.reshape(res_x, res_y, res_z)
    L = L.detach().cpu().numpy()
    
    grid = pv.ImageData()
    grid.dimensions = np.array(L.shape) + 1  # Add 1 because dimensions are number of points
    grid.spacing = (1, 1, 1)  # Voxel spacing
    grid.origin = (0, 0, 0)   # Origin of the grid

    # Add the density data to the grid as a cell array
    # Need to flatten the numpy array to match PyVista's expected format
    grid.cell_data["density"] = L.flatten(order="F")

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

    return L

# def plot_rays(pm: PhysicalModel, ro: np.ndarray, rd: np.ndarray, tn: np.ndarray, tf: np.ndarray):
#     p1 = ro + rd * tn[:, np.newaxis]
#     p2 = ro + rd * tf[:, np.newaxis]
#     p = np.concat([p1, p2], axis=0)
#     # print(p1.shape, p2.shape, p.shape)
    
#     ax = plt.figure().add_subplot(projection='3d')

#     for i in range(ro.shape[0]):
#         xs = [ro[i, 0], p2[i, 0]]
#         ys = [ro[i, 1], p2[i, 1]]
#         zs = [ro[i, 2], p2[i, 2]]
#         ax.plot(xs, ys, zs, color='b', linewidth=1)

#     x0, y0, z0 = ro[:, 0], ro[:, 1], ro[:, 2]
#     # x1, y1, z1 = p1[:, 0], p1[:, 1], p1[:, 2]
#     # x2, y2, z2 = p2[:, 0], p2[:, 1], p2[:, 2]
#     xp, yp, zp = p[:, 0], p[:, 1], p[:, 2]
#     ax.scatter(x0, y0, z0, color='g')
#     ax.scatter(xp, yp, zp, color='r')
#     # ax.scatter(x1, y1, z1, color='r')
#     # ax.scatter(x2, y2, z2, color='r')

#     vol = pm_desc.reconstruction_volume
#     frame = pm_desc.reference_frame
#     x_min, x_max = vol.oblique_range_x
#     y_min, y_max = vol.oblique_range_y
#     z_min, z_max = frame.origin_altitude, frame + vol.oblique_height
#     box_min = (x_min, y_min, z_min)
#     box_max = (x_max, y_max, z_max)
#     draw_bbox(ax, box_min, box_max)
    
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_aspect('equal', adjustable='box')
#     plt.tight_layout()
#     plt.show()

# def draw_bbox(ax, mins, maxs, color='cyan', alpha=0.2):
#     xmin, ymin, zmin = mins
#     xmax, ymax, zmax = maxs
#     # Define the 8 corners of the bounding box
#     corners = [
#         [xmin, ymin, zmin],
#         [xmax, ymin, zmin],
#         [xmax, ymax, zmin],
#         [xmin, ymax, zmin],
#         [xmin, ymin, zmax],
#         [xmax, ymin, zmax],
#         [xmax, ymax, zmax],
#         [xmin, ymax, zmax]
#     ]

#     # Define the 6 faces (as lists of corners)
#     faces = [
#         [corners[0], corners[1], corners[2], corners[3]],  # bottom
#         [corners[4], corners[5], corners[6], corners[7]],  # top
#         [corners[0], corners[1], corners[5], corners[4]],  # front
#         [corners[2], corners[3], corners[7], corners[6]],  # back
#         [corners[1], corners[2], corners[6], corners[5]],  # right
#         [corners[0], corners[3], corners[7], corners[4]]   # left
#     ]

#     # Create a 3D polygon collection and add it to the axis
#     bbox = Poly3DCollection(faces, linewidths=1, edgecolors='r', alpha=alpha)
#     bbox.set_facecolor(color)
#     ax.add_collection3d(bbox)
