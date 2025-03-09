from matplotlib import pyplot as plt
from pathlib import Path
from aurora.data import parse_matrix_data, parse_model_data

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
    model = parse_model_data(model_path)
    m, x, y = model.matrix, model.altitudes, model.energies
    plt.imshow(m, extent=[x[0], x[-1], y[-1], y[0]], origin="upper", aspect="auto")
    plt.colorbar()
    plt.xlabel("Altitude (z)")
    plt.ylabel("Energy (E)")
    plt.show()