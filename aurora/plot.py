from matplotlib import pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Callable
import torch

from aurora.camera import Camera
from aurora.utils import load_matrix_data

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
