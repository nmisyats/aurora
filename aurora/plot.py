from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pyvista as pv
from pathlib import Path
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Callable

from aurora.data import parse_matrix_data, Model, Camera

def plot_training_loss(loss: list[float]):
    plt.plot(loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training batch loss")
    plt.show()

def plot_camera_image(cam_path: Path):
    plt.imshow(parse_matrix_data(cam_path / "image.dat"))
    plt.colorbar()
    plt.title(cam_path.stem)
    plt.show()

def plot_camera_views(cam_path: Path):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), layout='compressed')
    
    im1 = ax1.imshow(parse_matrix_data(cam_path / "az_cam.dat"))
    ax1.set_title("azimuth")
    fig.colorbar(im1, orientation='vertical')

    im2 = ax2.imshow(parse_matrix_data(cam_path / "ze_cam.dat"))
    ax2.set_title("zenith")
    fig.colorbar(im2, orientation='vertical')

    plt.show()

def plot_model_matrix(model_path: Path):
    m = parse_matrix_data(model_path / "M_emis.dat")
    x = parse_matrix_data(model_path / "altitude.dat")
    y = parse_matrix_data(model_path / "energy.dat")
    plt.matshow(np.log(m))
    plt.colorbar()
    plt.ylabel("Altitude (z)")
    plt.xlabel("Energy (E)")
    plt.show()

def plot_total_energy_flux(estimate_f: Callable[[np.ndarray], np.ndarray], pm: Model, res_x: int, res_y: int):
    x = np.linspace(0.0, 1.0, res_x, dtype=np.float32)
    y = np.linspace(0.0, 1.0, res_y, dtype=np.float32)
    vol = pm.volume
    x = vol.min.x + x * (vol.max.x - vol.min.x)
    y = vol.min.y + y * (vol.max.y - vol.min.y)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    xy = np.stack((xx, yy), axis=-1)
    f = estimate_f(xy.reshape(res_x*res_y, 2))
    e = 1.602e-19
    lower_E, upper_E = pm.energy_bins[:-1], pm.energy_bins[1:]
    E = (lower_E + upper_E) / 2.0
    dE = upper_E - lower_E
    q = (10**3) * e * (10**4) * np.pi * (f * E * dE)
    q = np.sum(q, axis=1)
    q = q.reshape((res_x, res_y))

    if pm.q0 is not None:
        fig = plt.figure(figsize=(8, 4))
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(1, 2),
                        axes_pad=0.1,
                        cbar_location="right", cbar_mode="single", cbar_size="7%", cbar_pad="10%")
        
        vmin = min(q.min(), pm.q0.image.min())
        vmax = max(q.max(), pm.q0.image.max())

        x_min, x_max = pm.q0.min.x, pm.q0.max.x
        y_min, y_max = pm.q0.min.y, pm.q0.max.y
        grid[0].imshow(pm.q0.image,
                       interpolation='none',
                       extent=[y_min,y_max,x_max,x_min],
                       cmap="jet",
                       vmin=vmin, vmax=vmax)
        
        x_min, x_max = vol.min.x, vol.max.x
        y_min, y_max = vol.min.y, vol.max.y

        rect = patches.Rectangle((y_min, x_min), y_max - y_min, x_max - x_min, linewidth=1, edgecolor='r', facecolor='none')
        grid[0].add_patch(rect)
        
        im = grid[1].imshow(q,
                            interpolation='none',
                            extent=[y_min,y_max,x_max,x_min],
                            cmap="jet",
                            vmin=vmin, vmax=vmax)

        cbar = grid[0].cax.colorbar(im, cmap="jet")
        cbar.set_label("mW m$^{-2}$")

        grid[0].set_title("Reference $Q_0$")
        grid[1].set_title("Reconstructed $Q_0$")
        grid[0].set_xlabel("y (km)")
        grid[1].set_xlabel("y (km)")
        grid[0].set_ylabel("x (km)")
    else:
        x_min, x_max = vol.min.x, vol.max.x
        y_min, y_max = vol.min.y, vol.max.y
        plt.imshow(q, interpolation='none', extent=[y_min,y_max,x_max,x_min], cmap="jet")
        cbar = plt.colorbar(cmap="jet")
        cbar.set_label("mW m$^{-2}$")
        plt.xlabel("y (km)")
        plt.ylabel("x (km)")
        plt.title(f"Reconstructed total energy flux ({pm.name})")

    plt.show()

    return f

def plot_reconstructed_images(generate_image: Callable[[Camera], np.ndarray], cams: list[Camera]):
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
        img = generate_image(cams[i])
        ref = cams[i].image
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

def plot_volume_emission(estimate_L: Callable[[np.ndarray], np.ndarray], pm: Model, res_x: int, res_y: int, res_z: int):
    x = np.linspace(0.0, 1.0, res_x, dtype=np.float32)
    y = np.linspace(0.0, 1.0, res_y, dtype=np.float32)
    z = np.linspace(0.0, 1.0, res_z, dtype=np.float32)
    x = pm.box_min[0] + x * (pm.box_max[0] - pm.box_min[0])
    y = pm.box_min[1] + y * (pm.box_max[1] - pm.box_min[1])
    z = pm.box_min[2] + z * (pm.box_max[2] - pm.box_min[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    xyz = np.stack((xx, yy, zz), axis=-1)
    L = estimate_L(xyz.reshape(res_x*res_y*res_z, 3))
    L = L.reshape(res_x, res_y, res_z)
    
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

def plot_rays(pm: Model, ro: np.ndarray, rd: np.ndarray, tn: np.ndarray, tf: np.ndarray, o_ecef, to_spec_matrix):
    p1 = ro + rd * tn[:, np.newaxis]
    p2 = ro + rd * tf[:, np.newaxis]
    p = np.concat([p1, p2], axis=0)
    # print(p1.shape, p2.shape, p.shape)
    
    ax = plt.figure().add_subplot(projection='3d')

    for i in range(ro.shape[0]):
        xs = [ro[i, 0], p2[i, 0]]
        ys = [ro[i, 1], p2[i, 1]]
        zs = [ro[i, 2], p2[i, 2]]
        ax.plot(xs, ys, zs, color='b', linewidth=1)

    x0, y0, z0 = ro[:, 0], ro[:, 1], ro[:, 2]
    # x1, y1, z1 = p1[:, 0], p1[:, 1], p1[:, 2]
    # x2, y2, z2 = p2[:, 0], p2[:, 1], p2[:, 2]
    xp, yp, zp = p[:, 0], p[:, 1], p[:, 2]
    ax.scatter(x0, y0, z0, color='g')
    ax.scatter(xp, yp, zp, color='r')
    # ax.scatter(x1, y1, z1, color='r')
    # ax.scatter(x2, y2, z2, color='r')

    draw_bbox(ax, pm.box_min, pm.box_max)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def draw_bbox(ax, mins, maxs, color='cyan', alpha=0.2):
    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxs
    # Define the 8 corners of the bounding box
    corners = [
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax]
    ]

    # Define the 6 faces (as lists of corners)
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # bottom
        [corners[4], corners[5], corners[6], corners[7]],  # top
        [corners[0], corners[1], corners[5], corners[4]],  # front
        [corners[2], corners[3], corners[7], corners[6]],  # back
        [corners[1], corners[2], corners[6], corners[5]],  # right
        [corners[0], corners[3], corners[7], corners[4]]   # left
    ]

    # Create a 3D polygon collection and add it to the axis
    bbox = Poly3DCollection(faces, linewidths=1, edgecolors='r', alpha=alpha)
    bbox.set_facecolor(color)
    ax.add_collection3d(bbox)
