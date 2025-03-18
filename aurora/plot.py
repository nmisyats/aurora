from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from typing import Callable
from aurora.data import parse_matrix_data, Model, Camera

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
    x = np.linspace(0.0, 1.0, res_x, dtype=np.float32).reshape(1, res_x).repeat(res_y, axis=0)
    y = np.linspace(0.0, 1.0, res_y, dtype=np.float32).reshape(1, res_y).repeat(res_x, axis=0).T
    xy = np.stack((x, y), axis=-1)
    f = estimate_f(xy.reshape(res_x*res_y, 2))
    e = 1.602e-19
    lower_E, upper_E = pm.energy_bins[:-1], pm.energy_bins[1:]
    E = (lower_E + upper_E) / 2.0
    dE = upper_E - lower_E
    q = (10**3) * e * (10**4) * np.pi * (f * E * dE)
    q = np.sum(q, axis=1)
    q = q.reshape((res_x, res_y)).T

    if pm.has_ref_q0:
        fig = plt.figure(figsize=(8, 4))
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(1, 2),
                        axes_pad=0.1,
                        cbar_location="right", cbar_mode="single", cbar_size="7%", cbar_pad="10%")
        
        vmin = min(q.min(), pm.ref_q0_image.min())
        vmax = min(q.max(), pm.ref_q0_image.max())

        x_min, x_max = pm.ref_q0_range_min[0], pm.ref_q0_range_max[0]
        y_min, y_max = pm.ref_q0_range_min[1], pm.ref_q0_range_max[1]
        grid[0].imshow(pm.ref_q0_image,
                       interpolation='none',
                       extent=[y_min,y_max,x_max,x_min],
                       cmap="jet",
                       vmin=vmin, vmax=vmax)
        
        x_min, x_max = pm.box_min[0], pm.box_max[0]
        y_min, y_max = pm.box_min[1], pm.box_max[1]
        im = grid[1].imshow(q,
                            interpolation='none',
                            extent=[y_min,y_max,x_max,x_min],
                            cmap="jet",
                            vmin=vmin, vmax=vmax)

        cbar = grid[0].cax.colorbar(im, cmap="jet")
        cbar.set_label("mW m$^{-2}$")

        grid[0].set_title("Reference $Q_0$")
        grid[1].set_title("Reconstructed $Q_0$")
    else:
        x_min, x_max = pm.box_min[0], pm.box_max[0]
        y_min, y_max = pm.box_min[1], pm.box_max[1]
        plt.imshow(q, interpolation='none', extent=[y_min,y_max,x_max,x_min], cmap="jet")
        cbar = plt.colorbar(cmap="jet")
        cbar.set_label("mW m$^{-2}$")
        plt.xlabel("y (km)")
        plt.ylabel("x (km)")
        plt.title("Reconstructed total energy flux")

    plt.show()

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