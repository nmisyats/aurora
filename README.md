## Installation

Requires **3.11 or higher**. It is recommended to create a
[virtual environment](https://docs.python.org/3/library/venv.html)
and work within this environment:

```
python -m venv .venv
source .venv/bin/activate
```

Then, build and install the library `aurora` library and command line tool
by running the following commands at the root of the project directory:

```
pip install -r requirements.txt
pip install -e .
```

You can now use the `aurora` command line tool from the environment.
It has been dynamically installed, i.e. modifying the source code
modifies directly the installed `aurora` tool without need to rebuild
and reinstall.

To ensure installation is successfull, run the unit tests with:
```
python -m pytest tests
```

> **Note**: the tool may take a while to start the first time it is run.

## Command line usage

Adding `--help` to any command will display help information about this
command or its subcommands.

```
$ aurora --help

train   Train models
plot    Plotting utilities
gen     Generation utilities
```

### Training basics

To train a reconstruction, we must first define a configuration file
that describes the physical model that is used for reconstruction. A
template of such file, named `config.yaml` is shown below:

```yaml
frame: # Oblique reference frame description
  origin_lat: 69.348333333333
  origin_lon: 20.365000000000
  origin_alt: 90.0
  field_inc: 77.9
  field_dec: 6.0
bbox: # Reconstruction bounding box description
  east_min: -70.0
  east_max: 70.0
  south_min: -40.0
  south_max: 100.0
  alt_min: 90.0
  alt_max: 190.0
physics: # Physical model data
  emis_mat: ../model/M_emis.dat
  dens_mat: ../model/M_dens.dat
  altitude_bins: ../model/altitude.dat
  energy_bins: ../model/energy.dat
```

> **Note**: Paths used in YAML configuration files are relative to the *location of the YAML file*.

Training a reconstruction requires to first choose one of the
available model for the electron flux. The list of available
models can be obtained via `aurora train` command:
```
$ aurora train

Available models:
  spectral_mlp - MLP outputing the energy spectrum from the xy position
  spectral_res_mlp - Spectral MLP with a residual connection
  direct_mlp - MLP with position and energy input
  poly_mlp - MLP learning a polynomial basis of the flux

Use 'aurora train <model_name> --help' for model-specific options.
```

Each model share common arguments for training as well as
model-specific arguments. The command to train a model
`<model_name>` is:
```
$ aurora train <model_name>
```
For example, running `aurora train spectral_mlp --help` outputs:
```
Arguments
    config_file      PATH  Path to YAML configuration file [default: None] [required]

Options
    ---options                               PATH     Path to YAML training configuration file [default: None]
    --cam-pos                                PATH     Camera positions file [default: None]
    --cam-dir                                PATH     Cameras directory [default: None]
    --radar                                  PATH     Radar point cloud file [default: None]
    --gpu                  --no-gpu                   Use GPU if available [default: gpu]
    --iters                                  INTEGER  Number of training iterations [default: 2000]
    --ray-batch                              INTEGER  Batch size for ray loss [default: 4096]
    --ray-bins                               INTEGER  Number of bins for ray integration [default: 100]
    --radar-batch                            INTEGER  Batch size for radar loss [default: 1000]
    --smooth-batch                           INTEGER  Batch size for spectral smoothness loss [default: 1024]
    --ray-weight                             FLOAT    Weight for ray loss [default: 1.0]
    --radar-weight                           FLOAT    Weight for radar loss [default: 1.0]
    --smooth-weight                          FLOAT    Weight for spectral smoothness loss [default: 0.0]
    --lr                                     FLOAT    Initial learning rate [default: 5e-05]
    --reg-strength                           FLOAT    Regularization strength [default: 1.0]
    --lr-step                                INTEGER  Learning rate scheduler step [default: 1000]
    --lr-decay                               FLOAT    Learning rate step decay [default: 0.5]
    --save                                   PATH     Path to file where to save the reconstruction [default: None]
    --plot-loss            --no-plot-loss             Plot the training losses [default: no-plot-loss]
    --plot-flux            --no-plot-flux             Plot the reconstructed flux after training complete [default: plot-flux]
    --plot-res-x                             INTEGER  x resolution for plotting [default: 128]
    --plot-res-y                             INTEGER  y resolution for plotting [default: 128]
    --ref-flux                               PATH     Reference flux to compare the reconstruction with [default: None]
    --ref-config                             PATH     Path to configuration file for reference flux [default: None]
    --encoding-exp                           INTEGER  Fourier embedding maximum exponent [default: 4]
    --max-log_flux                           FLOAT    Logarithmic range of the flux [default: 7.0]
    --help                                            Show this message and exit.
```

### Example training

The most basic training on a set of cameras would be done as follows:
```
aurora train <model_name> path/to/config.yaml --cam-pos path/to/camera_position.dat --cam-dir path/to/camera/images --save recon.pth
```
The trained model will be saved as a self-contained file `recon.pth`.

If a reference flux is available to compare the reconsutrction (simulated data) with, it
can be added to the final flux plot by adding the following options:
```
--ref-flux path/to/flux.dat --ref-config path/to/flux_config.yaml
```

For convenience, it is possible to bundle training options into a YAML file
and load the options from it. For example:
```yaml
# train.yaml

cam_dir: path/to/camera/images
cam_pos: path/to/camera_position.set
iters: 5000
save: recon.pth
ref_flux: path/to/flux.dat
ref_config: path/to/flux_config.yaml
```
```
aurora train spectral_mlp config.yaml --options train.yaml
```

Missing options from the training file will use their default value.
It is still possible to add manual options to the command line and override
options set in the file. For example:
```
aurora train spectral_mlp config.yaml --options train.yaml --iters 2000 --lr 1e-4
```

### Generating and plotting data

`aurora plot` contains utility commands for plotting data that is static
or has already been generated. To generate new data (total flux, emission rate, images)
from a pretrained reconstruction (or reference flux), use `aurora gen`.

For example, to generate, plot, and save a 3D emission volume rate from a pretrained
`recon.pth` reconstruction, use:
```
aurora gen emis recon.pth --save emis_rate.dat
```
This will both save the generated emission and plot it for vizualization.
To vizualize the generated emission rate later, use:
```
aurora plot emis emis_rate.dat config.yaml
```

> **Note**: it is required to provide the physical configuration to be used for
vizualizing the generated data. Generated data only saves its raw content,
unlike reconstruction models which bundles also physical configuration.


## Library usage

### Minimal example

The following python script illustrates how to use `aurora`
as a python library. This is especially useful to easily create more
advanced custom experiments.

```python
# example.py

import torch
from matplotlib import pyplot as plt

from aurora import Reconstruction, save_reconstruction
from aurora import Frame, BBox
from aurora import train, RayLoss, SpectralSmoothnessLoss
from aurora.dataset import CameraRaysDataset
from aurora.models import SpectralMLP
import aurora.data as data
import aurora.plot as aplt
from aurora.utils import xy_grid

# Choose a device to run the reconstruction on
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Define the oblique reference frame
frame = Frame(
    origin_latitude=69.348333333333,
    origin_longitude=20.365000000000,
    origin_altitude=90.0,
    field_inclination=77.9,
    field_declination=6.0,
    device=device
)
print(frame)

# Define the reconstruction bounding box
bbox = BBox(
    frame=frame,
    south_range=(-40.0, 100.0),
    east_range=(-70.0, 70.0),
    altitude_range=(90.0, 190.0)
)
print(bbox)

# Load physical model data
altitude_bins = data.load_altitude_bins("../model/altitude.dat").to(device)
energy_bins = data.load_energy_bins("../model/energy.dat").to(device)
emis_mat = data.load_emission_matrix("../model/M_emis.dat").to(device)
dens_mat = data.load_density_matrix("../model/M_dens.dat").to(device)

# Instantiate the trainable flux model
flux_model = SpectralMLP(
    xy_min=bbox.xy_min,
    xy_max=bbox.xy_max,
    energy_bins=energy_bins
).to(device)
print(flux_model)

# Create te reconstruction using the flux model
recon = Reconstruction(
    flux_model=flux_model,
    frame=frame,
    bbox=bbox,
    emis_mat=emis_mat,
    dens_mat=dens_mat,
    altitude_bins=altitude_bins
)

# Load the cameras dataset and preprocess ray data
cameras = data.load_cameras("../datasets/camera_position.set", "../datasets/simulation1")
ray_data = CameraRaysDataset(cameras, frame, bbox)

# Define loss terms for training
loss_terms = [
    RayLoss(ray_data, batch_size=4096),
    SpectralSmoothnessLoss(batch_size=1024, weight=0.001),
]
# Train the reconstruction
train(recon, loss_terms, iters=2000)
# Save the reconstruction after training
save_reconstruction(recon, "./example.pth")

# Plot the reconstructed total energy flux

# Set the reconstruction in evaluation mode
# (this disables gradient computation for more efficiency)
recon.eval()
# Create a uniform grid spanning the xy bounding box
xy = xy_grid(bbox.xy_min, bbox.xy_max, 128, 128)
# Plot the flux
aplt.plot_flux_2d(
    flux_data=recon.flux(xy),
    xy_bounds=(bbox.xy_min, bbox.xy_max),
    energy_edges=energy_bins
)
plt.show()
```
Run the script directly with
```
python example.py
```

### Defining custom flux models

To create a custom flux model to use for training a reconstruction,
it is simply a matter of defining a new class that inherits from
the `FluxModel` base class in `aurora.models`, and fill
the required abstract properties and methods. A rough template
is shown below.

```python
from aurora.models import FluxModel

class MyFluxModel(FluxModel):
    def __init__(self, xy_min, xy_max, energy_bins, ...): # Custom construction
        super().__init__(xy_min, xy_max, energy_bins)
        ... # Custom initialization

    def forward(self, xy: torch.Tensor):
        # Pytorch's nn.Module forward method outputing the flux estimate
        # input: (N, 2) xy tensor
        # output: (N, n_bins) tensor
        return ...
```
It can then be used when instantiating a reconstruction as follows:
```python
recon = Reconstruction(
    flux_model=MyFluxModel(...),
    frame=frame,
    bbox=bbox,
    emis_mat=emis_mat,
    dens_mat=dens_mat,
    altitude_bins=altitude_bins
)
```

In order to add a custom model to the `aurora train` command,
the custom model class **must** be defined in the `aurora/models.py`
file. It must also follow a specific format as described below:
```python
# Define extra command line arguments with their default values
# and help string. Must be defined even if empty
@dataclass
class MyModelConfig:
    param1: int = config_field(42, help="Custom param 1")
    param2: float = config_field(3.14, help="Custom param 2")

# Define and register model with its name in the CLI
@register_model("my_model", MyModelConfig)
class MyModel(TrainableFluxModel):
    "Short description of model" # Shown in the `aurora train` command

    def __init__(
            self,
            # Mandatory argument
            xy_min: torch.Tensor,
            xy_max: torch.Tensor,
            energy_bins: torch.Tensor,
            # Command line arguments with default
            config: MyModelConfig = MyModelConfig()
        ):
        super().__init__(xy_min, xy_max, energy_bins)
        ...

    def forward(self, xy: torch.Tensor):
        ...
```
It can be then be trained as any other model using its registered name, with
custom extra arguments:
```
aurora train my_model path/to/config.yaml --cam-pos path/to/camera_position.dat --cam-dir path/to/camera/images --save recon.pth --param1 74 --param2 2.718
```