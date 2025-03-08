from matplotlib import pyplot as plt
from pathlib import Path
from aurora.data import parse_image_data

def plot_camera_image(cam_path: Path):
    plt.imshow(parse_image_data(cam_path / "image.dat"))
    plt.colorbar()
    plt.title(cam_path.stem)
    plt.show()

def plot_camera_views(cam_path: Path):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), layout='compressed')
    
    im1 = ax1.imshow(parse_image_data(cam_path / "az_cam.dat"))
    ax1.set_title("azimuth")
    fig.colorbar(im1, orientation='vertical')

    im2 = ax2.imshow(parse_image_data(cam_path / "ze_cam.dat"))
    ax2.set_title("zenith")
    fig.colorbar(im2, orientation='vertical')

    plt.show()