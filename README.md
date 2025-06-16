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
modifies directly the installed `aurroa` tool without need to rebuild
and reinstall.

To ensure installation is successfull, run the unit tests with:
```
python -m pytest tests
```

**Note**: the tool may take a while to start the first time it is run.

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
  altitudes: ../model/altitude.dat
  energies: ../model/energy.dat
```

Training a reconstruction requires to first choose one of the
available model for the electron flux. The list of available
models can be obtained via `aurora train` command:
```
$ aurora train

Available models:
  log_mlp
  log_res_mlp
```

Each model share common arguments for training as well as
model-specific arguments. The command to train a model
`<model_name>` is:
```
$ aurora train <model_name>
```
For example, running `--help` for `log_mlp` outputs:
```
Arguments
    config_file      PATH  Path to YAML configuration file [default: None] [required]

Options
    --training-options                       PATH     Path to YAML training configuration file [default: None]
    --cam-pos                                PATH     Camera positions file [default: None]
    --cam-dir                                PATH     Cameras directory [default: None]
    --radar-points                           PATH     Radar point cloud file [default: None]
    --gpu                  --no-gpu                   Use GPU if available [default: gpu]
    --iters                                  INTEGER  Number of training iterations [default: 2000]
    --ray-batch-size                         INTEGER  Batch size for ray loss [default: 4096]
    --ray-bins                               INTEGER  Number of bins for ray integration [default: 100]
    --radar-batch-size                       INTEGER  Batch size for radar loss [default: 1000]
    --ray-loss-weight                        FLOAT    Weight for ray loss [default: 1.0]
    --radar-loss-weight                      FLOAT    Weight for radar loss [default: 1.0]
    --lr                                     FLOAT    Initial learning rate [default: 5e-05]
    --reg-strength                           FLOAT    Regularization strength [default: 1.0]
    --lr-step                                INTEGER  Learning rate scheduler step [default: 1000]
    --lr-decay                               FLOAT    Learning rate step decay [default: 0.5]
    --save                                   PATH     Path to file where to save the reconstruction [default: None]
    --plot-loss            --no-plot-loss             Plot the training losses [default: no-plot-loss]
    --plot-flux            --no-plot-flux             Plot the reconstructed flux after training complete [default: plot-flux]
    --plot-res-x                             INTEGER  x resolution for plotting [default: 128]
    --plot-res-y                             INTEGER  y resolution for plotting [default: 128]
    --ref-flux-data                          PATH     Reference flux to compare the reconstruction with [default: None]
    --ref-flux-config                        PATH     Path to configuration file for reference flux [default: None]
    --encoding-exp                           INTEGER  Fourier embedding maximum exponent [default: 4]
    --log-scale                              FLOAT    Logarithmic range of the flux [default: 7.0]
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
--ref-flux-data path/to/flux.dat --ref-flux-config path/to/flux_config.yaml
```

For convenience, it is possible to bundle training options into a YAML file
and load the options from it. For example:
```yaml
# train.yaml

cam_dir: path/to/camera/images
cam_pos: path/to/camera_position.set
iters: 5000
save: recon.pth
ref_flux_data: path/to/flux.dat
ref_flux_config: path/to/flux_config.yaml
```
```
aurora train log_mlp config.yaml --training-options train.yaml
```

Missing options from the training file will use their default value.
It is still possible to add manual options to the command line and override
options set in the file. For example:
```
aurora train log_mlp config.yaml --training-options train.yaml --iters 500 --lr 1e-4
```

### Generating and plotting data

`aurora plot` contains utility commands for plotting data that is static
or has already been generated. To generate new data (total flux, emission rate, images)
from a pretrained reconstruction, use `aurora gen`.

For example, to generate plot, and save a 3D emission rate volume from a pretrained
`recon.pth` model, use:
```
aurora gen emis recon.pth --save emis_rate.dat
```
This will both save the generated emission and plot it for vizualization.
To vizualize the generated emission rate later use:
```
aurora plot emis emis_rate.dat config.yaml
```

**Note**: it is required to provide the physical configuration to be used for
vizualizing the generated data. Generated data only saves its raw content,
unlike reconstruction models which bundles also physical information.


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
from aurora.dataset import CameraRaysDataset
from aurora.models import LogMLP
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
z_edges = data.load_altitude_bins("../model/altitude.dat").to(device)
E_edges = data.load_energy_bins("../model/energy.dat").to(device)
M_emis = data.load_emission_matrix("../model/M_emis.dat").to(device)
M_dens = data.load_density_matrix("../model/M_dens.dat").to(device)

# Instantiate the trainable flux model
flux_model = LogMLP(
    xy_min=bbox.xy_min,
    xy_max=bbox.xy_max,
    E_edges=E_edges,
    config=LogMLP.default_config()
).to(device)
print(flux_model)

# Create te reconstruction using the flux model
recon = Reconstruction(
    flux_model=flux_model,
    frame=frame,
    bbox=bbox,
    M_emis=M_emis,
    M_dens=M_dens,
    z_edges=z_edges
)

# Load the cameras dataset and preprocess ray data
cameras = data.load_cameras("../datasets/camera_position.set", "../datasets/simulation1")
ray_data = CameraRaysDataset(cameras, frame, bbox)

# Train the reconstruction
recon.train(ray_data=ray_data, num_iters=2000)
# Save the reconstruction after training
save_reconstruction(recon, "./recon.pth")

# Plot the reconstructed total energy flux

# Set the reconstruction in evaluation mode
# (this disables gradient computation for more efficiency)
recon.eval_mode()
# Create a uniform grid spanning the xy bounding box
xy = xy_grid(bbox.xy_min, bbox.xy_max, 128, 128)
# Plot the flux
aplt.plot_flux_2d(
    flux_data=recon.flux(xy),
    xy_bounds=(bbox.xy_min, bbox.xy_max),
    energy_edges=E_edges
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
the `TrainableFluxModel` base class in `aurora.models`, and fill
the required abstract properties and methods. A rough template
is shown below.

```python
from aurora.models import TrainableFluxModel

class MyFluxModel(TrainableFluxModel):
    def __init__(self, ...): # Custom constructioon
        super().__init__()
        ... # Custom initialization

    def forward(self, xy: torch.Tensor):
        # Pytorch's nn.Module forward method outputing the flux estimate
        # input: (N, 2) xy tensor
        # output: (N, n_bins) tensor
        return ...
    
    @property
    def xy_min(self):
        # output: (2,) tensor, lower xy bounds
        return ...
    
    @property
    def xy_max(self):
        # output: (2,) tensor, upper xy bounds
        return ...
    
    @property
    def E_edges(self):
        # output: (n_bins + 1,) tensor, energy bins edges
        return ...
```
It can then be used when instantiating a reconstruction as follows:
```python
recon = Reconstruction(
    flux_model=MyFluxModel(...),
    frame=frame,
    bbox=bbox,
    M_emis=M_emis,
    M_dens=M_dens,
    z_edges=z_edges
)
```

In order to add a custom model to the `aurora train` command
line, the custom model class **must** be defined in the `aurora/models.py`
file. It must also follow a specific format as described below:
```python
# Define extra command line arguments with their default values
# and help string
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
            E_edges: torch.Tensor,
            # Command line arguments
            config: MyModelConfig
        ):
        super().__init__()
        ...

    def forward(self, xy: torch.Tensor):
        ...
    
    @property
    def xy_min(self):
        return ...
    
    @property
    def xy_max(self):
        return ...
    
    @property
    def E_edges(self):
        return ...

    # Optional, useful when using as library
    @classmethod
    def default_config(cls):
        return MyModelConfig()
```
It can be then be trained as any other model using its registered name, with
custom extra arguments:
```
aurora train my_model path/to/config.yaml --cam-pos path/to/camera_position.dat --cam-dir path/to/camera/images --save recon.pth --param1 74 --param2 2.718
```