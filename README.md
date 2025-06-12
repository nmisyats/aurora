# Installation

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

# Command line usage

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
bbox: # Rconstruction bounding box description
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
ref-flux-data path/to/flux.dat ref-flux-config path/to/flux_config.yaml
```

For convenience, it is possible to bundle training options into a YAML file
and load the options from it. For example:
```yaml
# train.yaml

cam_dir: path/to/camera/images
cam_pos: path/to/camera_position.dat
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