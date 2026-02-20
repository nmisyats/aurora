## Installation

Requires **Python 3.11 or higher**. It is recommended to create a
[virtual environment](https://docs.python.org/3/library/venv.html)
and work within this environment:

**Linux:**

```
python -m venv .venv
source .venv/bin/activate
```

**Windows:**

```
python -m venv .venv
.\.venv\Scripts\activate
```

Then, build and install the `aurora` library and command line tool
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

To train a reconstruction, first define a configuration file
that describes the physical model that is used for reconstruction.
An example `config.yaml` is shown below:

```yaml
frame: # Oblique reference frame
  origin_lat: 69.3483
  origin_lon: 20.3650
  origin_alt: 90.0
  field_inc: 77.9
  field_dec: 6.0
bbox: # Reconstruction bounding box
  east_min: -70.0
  east_max: 70.0
  south_min: -40.0
  south_max: 100.0
  alt_min: 90.0
  alt_max: 190.0
physics: # Physical model data
  emis_mat: model/M_emis.dat
  dens_mat: model/M_dens.dat
  altitude_bins: model/altitude.dat
  energy_bins: model/energy.dat
```

> **Note**: Paths provided in YAML configuration files are relative to the *location of the YAML file*, unless provided as absolute paths.

Training a reconstruction requires to first choose one of the
available model for the electron flux. The list of available
models can be obtained via `aurora train` command:
```
$ aurora train

Available models:
  spectral_mlp - MLP outputing the energy spectrum from the xy position
  poly_mlp - MLP learning a polynomial basis of the flux
  hybrid_mlp - Hybrid model using two MLPs for position and energy embedding

Use 'aurora train <model_name> --help' for model-specific options.
```

Each model share common arguments for training as well as
model-specific arguments. The command to train a model
`<model_name>` is `aurora train <model_name>`.
For example, running `aurora train spectral_mlp --help` outputs:
```
Usage: aurora train spectral_mlp [OPTIONS] CONFIG_FILE

  Train spectral_mlp model

Arguments:
  CONFIG_FILE  Path to YAML configuration file  [required]

Options:
  --options PATH                Path to YAML training configuration file
  --cams PATH                   Camera data directory containing
                                camera_position.set and directory for each
                                camera
  --radar PATH                  Radar point cloud file
  --gpu / --no-gpu              Use GPU if available  [default: gpu]
  --iters INTEGER               Number of training iterations  [default: 2000]
  --ray-batch INTEGER           Batch size for ray loss  [default: 4096]
  --ray-bins INTEGER            Number of bins for ray integration  [default:
                                100]
  --radar-batch INTEGER         Batch size for radar loss  [default: 1024]
  --smooth-batch INTEGER        Batch size for spectral smoothness loss
                                [default: 1024]
  --ray-weight FLOAT            Weight for ray loss  [default: 1.0]
  --radar-weight FLOAT          Weight for radar loss  [default: 1.0]
  --smooth-weight FLOAT         Weight for spectral smoothness loss  [default:
                                0.0]
  --lr FLOAT                    Initial learning rate  [default: 5e-05]
  --reg-strength FLOAT          Parameter L2 regularization strength
                                [default: 1.0]
  --save PATH                   Path to file where to save the reconstruction
  --plot-loss / --no-plot-loss  Plot the training losses  [default: no-plot-
                                loss]
  --plot-flux / --no-plot-flux  Plot the reconstructed flux after training
                                complete  [default: plot-flux]
  --plot-res-x INTEGER          x resolution for plotting  [default: 128]
  --plot-res-y INTEGER          y resolution for plotting  [default: 128]
  --ref PATH                    Path to folder containing reference flux.dat
                                and config.yaml to compare with
  --enc-exp INTEGER             Maximum positional encoding exponent
                                [default: 4]
  --max-log-f INTEGER           Maximum logarithmic value of the reconstructed
                                flux  [default: 7.0]
  --num-hidden INTEGER          Number of hidden layers  [default: 4]
  --hidden-size INTEGER         Size of each hidden layer  [default: 128]
  --help                        Show this message and exit.
```

### Example training

The most basic training on a set of cameras would be done as follows:
```
aurora train <model_name> path/to/config.yaml --cams path/to/camera/dataset --save model.pth
```
The trained model will be saved as a self-contained file `model.pth`.

If a ground truth flux is available to compare the reconstruction with, it
can be added to the final flux plot by adding the path to a directory containing
a `flux.dat` and `config.yaml` files with the options:
```
--ref path/to/groundtruth
```

For convenience, it is possible to bundle training options into a YAML file
and load the options from it. For example:
```yaml
# train.yaml

cams: path/to/camera/dataset
iters: 5000
save: model.pth
ref: path/to/groundtruth
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
or has already been generated into a `.dat` file. To generate new data (total flux, emission rate, images)
from a pretrained reconstruction (or reference flux), use `aurora gen`.

For example, to generate, plot, and save a 3D emission volume rate from a pretrained
`model.pth` reconstruction, use:
```
aurora gen emis model.pth --save emis_rate.dat
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
as a python library.

```python
# example.py

import torch
from matplotlib import pyplot as plt

import aurora as au
from aurora import Frame, BBox, ModelConfig, RayDataset, StratifiedSampler
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
)
# Define the reconstruction bounding box
bbox = BBox(
    frame=frame,
    south_range=(-40.0, 100.0),
    east_range=(-70.0, 70.0),
    altitude_range=(90.0, 190.0)
)
# Create the physical model configuration
config = ModelConfig(
    bbox=bbox,
    altitude_bins=au.data.load_altitude_bins("model/altitude.dat"),
    energy_bins=au.data.load_energy_bins("model/energy.dat"),
    emis_mat=au.data.load_emission_matrix("model/M_emis.dat"),
    dens_mat=au.data.load_density_matrix("model/M_dens.dat")
)

# Instantiate the trainable flux model on the chosen device
model = SpectralMLP(config).to(device)
print(model)

# Load the cameras images and preprocess ray data
cameras = au.data.load_cameras("./dataset")
ray_data = RayDataset(cameras, bbox.expand(), device) # Inifinitely wide bounding box

# Define the loss function that evaluates a model's loss during one
# training iteration
ray_sampler = StratifiedSampler(num_bins=64) # Ray sampler for training
def iter_loss(model):
    batch = ray_data.sample_batch(4096) # Random batch of 4096 rays
    loss = au.ray_loss(model, batch, ray_sampler) # Evaluate the loss
    return loss

# Train the model with the defined loss for 5000 iterations
au.train(model, iter_loss, 5000)

# Save the reconstruction after training
au.save_model(model, "./example.pth")

# Plot the reconstructed total energy flux
model.eval()
with torch.no_grad():
    # Create a uniform grid spanning the xy bounding box, generate and
    # plot the total energy flux
    xy = bbox.xy_grid(100, 100).to(device)
    q0 = model.total_energy_flux(xy)
    au.plot.plot_flux_2d(q0, bbox.xy_bounds)
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
the required abstract forward method. A rough template
is shown below.

```python
from aurora.models import FluxModel

class MyModel(FluxModel):
    def __init__(
            self,
            config: ModelConfig,
            ... # Custom arguments
        ):
        super().__init__(config)
        ... # Custom initialization

    def forward(self, xy: torch.Tensor):
        # Pytorch's nn.Module forward method outputing the flux estimate
        # input: (N, 2) xy tensor, normalized in [0, 1]
        # output: dictionary containing at least an "f" entry
        return {
            "f": ... # (N, n_energy_bins)
        }
```

### Adding a model to the command line

To add a custom model to the `aurora train` command, you must
first define your new model:
```python
class MyModel(FluxModel):
    def __init__(
            self,
            config: ModelConfig,
            # Extra arguments
            param1: int = 42,
            param2: float = 3.14
        ):
        super().__init__(config)
        ...

    def forward(self, xy: torch.Tensor):
        ...
```
Then, define its training command at the bottom of the `aurora_cli/train_app.py`
using the following template:
```python
... # Other models

@model_train_command("my_model", "My model description")
def train_hybrid_mlp(
    config: models.ModelConfig,
    training_config: TrainingConfig,
    param1: int = typer.Option(42, help="Help for param1"),
    param2: float = typer.Option(3.14, help="Help for param2"),
):
    # Instantiate reconstruction model
    model = models.HybridMLP(
        config=config,
        # Model-specific
        param1=param1,
        param2=param2
    ).to(training_config.device)
    # Print model details (optional)
    typer.echo(f"Instantiated model:\n{model}") 
    # Train the reconstruction on the provided data using
    # the default predefined loss
    history = au.train(
        model=model,
        iter_loss=default_iter_loss(training_config),
        num_iters=training_config.iters,
        lr=training_config.lr,
        weight_decay=training_config.reg_strength,
    )
    return model, history
```
This will add a new subcommand to `aurora train` with the provided extra options:
```
$ aurora train

Available models:
  ...
  my_model - My model description

$ aurora train my_model --help

Arguments
    config_file      PATH  Path to YAML configuration file [default: None] [required]

Options
    ...
    --param1                                 INTEGER  Custom param1 [default: 42]
    --param2                                 FLOAT    Custom param2 [default: 3.14]
    ...
```
It can be then be trained as any other model using its registered name, with
custom extra arguments:
```
aurora train my_model path/to/config.yaml --cams path/to/camera/dataset --save my_model.pth --param1 74 --param2 2.718
```