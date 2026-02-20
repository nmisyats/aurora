from pathlib import Path
from typing import Optional, Dict, TYPE_CHECKING

from schema import Schema, Optional as Opt, And, Or, Use
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from .common import PathLike
from .physics import (
    load_altitude_bins,
    load_energy_bins,
    load_emission_matrix,
    load_density_matrix,
)
from aurora.frame import Frame
from aurora.bbox import BBox

if TYPE_CHECKING:
    from aurora.models.flux_model import ModelConfig


def load_yaml(file_path: PathLike) -> Dict:
    with open(file_path, "r") as f:
        data = yaml.load(f, Loader=Loader)
    return data


def save_yaml(file_path: PathLike, data):
    with open(file_path, "w") as f:
        yaml.dump(data, f, Dumper=Dumper)


def resolve_relative_to_base(target_path: PathLike, base_path: PathLike) -> Path:
    """
    Return an absolute, resolved Path for `target_path`; if it is relative, interpret
    it relative to `base_path`.
    """
    target_path = Path(target_path)
    if target_path.is_absolute():
        return target_path.resolve()
    base_path = Path(base_path)
    return (base_path.resolve() / target_path).resolve()


def path_validator(base_path: Optional[PathLike] = None):
    """
    Creates schema validator that ensures a given path exists. If `base_path` is
    provided, relative paths are resolved against it.
    """
    validators = [Use(Path)]
    if base_path is not None:
        validators.append(Use(lambda p: resolve_relative_to_base(p, base_path)))
    validators.append(lambda p: p.exists())
    return And(*validators)


def config_schema(base_path=None):
    valid_path = path_validator(base_path)
    return Schema({
        "frame": {
            "origin_lat": Use(float),
            "origin_lon": Use(float),
            "origin_alt": Use(float),
            "field_inc": Use(float),
            "field_dec": Use(float),
        },
        "bbox": {
            "east_min": Use(float),
            "east_max": Use(float),
            "south_min": Use(float),
            "south_max": Use(float),
            "alt_min": Use(float),
            "alt_max": Use(float),
        },
        "physics": {
            Opt("emis_mat"): Or(valid_path, {str: valid_path}, only_one=True),
            Opt("dens_mat"): valid_path,
            "altitude_bins": valid_path,
            "energy_bins": valid_path,
        },
    }, ignore_extra_keys=True)


def load_config(yaml_path: PathLike):
    yaml_path = Path(yaml_path)
    schema = config_schema(yaml_path.parent)
    data = load_yaml(yaml_path)
    data = schema.validate(data)
    return config_from_dict(data)


def config_from_dict(data: dict) -> "ModelConfig":
    from aurora.models.flux_model import ModelConfig

    frame_data = data["frame"]
    frame = Frame(
        origin_latitude=frame_data["origin_lat"],
        origin_longitude=frame_data["origin_lon"],
        origin_altitude=frame_data["origin_alt"],
        field_inclination=frame_data["field_inc"],
        field_declination=frame_data["field_dec"],
    )

    bbox_data = data["bbox"]
    bbox = BBox(
        south_range=(bbox_data["south_min"], bbox_data["south_max"]),
        east_range=(bbox_data["east_min"], bbox_data["east_max"]),
        altitude_range=(bbox_data["alt_min"], bbox_data["alt_max"]),
        frame=frame,
    )

    phys_data = data["physics"]
    altitude_bins = load_altitude_bins(phys_data["altitude_bins"])
    energy_bins = load_energy_bins(phys_data["energy_bins"])

    emis_mats = None
    if "emis_mat" in phys_data:
        if isinstance(phys_data["emis_mat"], Path):
            emis_mats = load_emission_matrix(phys_data["emis_mat"])
        else:
            emis_mats = {}
            for wl, emis_dat_path in phys_data["emis_mat"].items():
                emis_mats[wl] = load_emission_matrix(emis_dat_path)

    dens_mat = None
    if "dens_mat" in phys_data:
        dens_mat = load_density_matrix(phys_data["dens_mat"])

    return ModelConfig(
        bbox=bbox,
        altitude_bins=altitude_bins,
        energy_bins=energy_bins,
        emis_mat=emis_mats,
        dens_mat=dens_mat,
    )

