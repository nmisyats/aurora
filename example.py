# example.py

import torch
from matplotlib import pyplot as plt

import aurora as au
from aurora import Frame, BBox, StratifiedSampler
from aurora.models import SpectralMLP

# Choose a device to run the reconstruction on
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Define the oblique reference frame
frame = Frame(
    origin_latitude=69.348333333333,
    origin_longitude=20.365000000000,
    origin_altitude=90.0,
    field_inclination=77.9,
    field_declination=6.0
).to(device)
print(frame)

# Define the reconstruction bounding box
bbox = BBox(
    frame=frame,
    south_range=(-40.0, 100.0),
    east_range=(-70.0, 70.0),
    altitude_range=(90.0, 190.0)
).to(device)
print(bbox)

# Load physical model data
altitude_bins = au.data.load_altitude_bins("../model/altitude.dat")
energy_bins = au.data.load_energy_bins("../model/energy.dat")
emis_mat = au.data.load_emission_matrix("../model/M_emis.dat")
dens_mat = au.data.load_density_matrix("../model/M_dens.dat")

# Instantiate the trainable flux model in the chosen device
model = SpectralMLP(
    frame=frame,
    bbox=bbox,
    emis_mat=emis_mat,
    dens_mat=dens_mat,
    altitude_bins=altitude_bins,
    energy_bins=energy_bins
).to(device)
print(model)

# Load the cameras images and preprocess ray data
cameras = au.data.load_cameras("../datasets/camera_position.set", "../datasets/simulation1")
ray_data = au.datasets.RayDataset(cameras, frame, bbox).to(device)

# Choose a ray sampler for training
ray_sampler = StratifiedSampler(num_bins=64)
# Implement training iteration step
def train_step():
    batch = ray_data.sample_batch(4096) # random batch of 4096 rays
    loss = au.ray_loss(model, batch, ray_sampler) # evaluate the loss
    loss.backward() # optimize
    return {"ray_loss": loss.item()}

# Train the model with the given training step for 2000 iterations
au.train(modules=model, step=train_step, num_iters=2000)
# Save the reconstruction after training
au.save_model(model, "./example.pth")

## Plot the reconstructed total energy flux

# Set the reconstruction in evaluation mode
# (this disables gradient computation for more efficiency)
model.eval()
# Create a uniform grid spanning the xy bounding box
xy = au.utils.xy_grid(bbox.xy_min, bbox.xy_max, 128, 128).to(device)
# Plot the flux
au.plot.plot_flux_2d(
    flux_data=model.flux(xy).cpu(),
    xy_bounds=(bbox.xy_min, bbox.xy_max),
    energy_edges=energy_bins
)
plt.show()