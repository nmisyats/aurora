from typing import Union, Tuple, List
import functools
import inspect

import numpy as np
import torch

Number = int | float
TensorLike = Union[torch.Tensor, np.ndarray, Number, List[Number], Tuple[Number, ...]]

def as_tensor(data: TensorLike) -> torch.Tensor:
    """
    Converts input data to a PyTorch tensor.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data)
    elif isinstance(data, (float, int)):
        return torch.tensor(data)
    else:
        raise TypeError(f"Unsupported type for conversion to tensor: {type(data)}")

Coord2DLike = Union[Tuple[float, float], List[float], torch.Tensor]
Coord3DLike = Union[Tuple[float, float, float], List[float], torch.Tensor]

def bounds2d_to_tuple(min_point: Coord2DLike, max_point: Coord2DLike) -> Tuple[float, float, float, float]:
    """
    Converts ((x_min, y_min), (x_max, y_max)) to (x_min, x_max, y_min, y_max).
    """
    x_min, y_min = float(min_point[0]), float(min_point[1])
    x_max, y_max = float(max_point[0]), float(max_point[1])
    return x_min, x_max, y_min, y_max

def ranges2d_to_tuple(x_range: Coord2DLike, y_range: Coord2DLike) -> Tuple[float, float, float, float]:
    """
    Converts ((x_min, x_max), (y_min, y_max)) to (x_min, x_max, y_min, y_max).
    """
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])
    return x_min, x_max, y_min, y_max

def bounds3d_to_tuple(min_point: Coord3DLike, max_point: Coord3DLike) -> Tuple[float, float, float, float, float, float]:
    """
    Converts ((x_min, y_min, z_min), (x_max, y_max, z_max)) to (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    x_min, y_min, z_min = float(min_point[0]), float(min_point[1]), float(min_point[2])
    x_max, y_max, z_max = float(max_point[0]), float(max_point[1]), float(max_point[2])
    return x_min, x_max, y_min, y_max, z_min, z_max

def ranges3d_to_tuple(x_range: Coord2DLike, y_range: Coord2DLike, z_range: Coord2DLike) -> Tuple[float, float, float, float, float, float]:
    """
    Converts ((x_min, x_max), (y_min, y_max), (z_min, z_max)) to (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])
    z_min, z_max = float(z_range[0]), float(z_range[1])
    return x_min, x_max, y_min, y_max, z_min, z_max

def xy_grid(
        xy_min: torch.Tensor,
        xy_max: torch.Tensor,
        res_x: int,
        res_y: int,
        device: torch.device | None = None
    ) -> torch.Tensor:
    if device is None:
        device = xy_min.device
    x = torch.linspace(xy_min[0], xy_max[0], res_x, device=device)
    y = torch.linspace(xy_min[1], xy_max[1], res_y, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack((xx, yy), dim=-1)
    return xy

def xyz_grid(
        xyz_min: torch.Tensor,
        xyz_max: torch.Tensor,
        res_x: int,
        res_y: int,
        res_z: int,
        device: torch.device | None = None
    ) -> torch.Tensor:
    if device is None:
        device = xyz_min.device
    x = torch.linspace(xyz_min[0], xyz_max[0], res_x, device=device)
    y = torch.linspace(xyz_min[1], xyz_max[1], res_y, device=device)
    z = torch.linspace(xyz_min[2], xyz_max[2], res_z, device=device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    xyz = torch.stack((xx, yy, zz), dim=-1)
    return xyz

def downsample_image(image: torch.Tensor, factor: int) -> np.ndarray:
    """
    Downsamples a 2D or 3D image (e.g., grayscale or RGB) by picking every `factor`-th pixel.
    """
    if factor < 1:
        raise ValueError("Downsampling factor must be >= 1")
    
    # Handle 2D (grayscale) or 3D (color) images
    if image.ndim == 2:
        return image[::factor, ::factor]
    else:
        return image[::factor, ::factor, :]

def ceiled_div(a: int, b: int):
    return -(-a // b) # ceil(numel/chunk_size)

def iter_chunks(numel: int, chunk_size: int):
    num_chunks = ceiled_div(numel, chunk_size)
    for i in range(num_chunks):
        chunk_start = i * chunk_size
        chunk_stop = min((i+1) * chunk_size, numel)
        yield chunk_start, chunk_stop

def normalize_batch_dims(
    ndim_in: dict[str, int],
    ndim_out: int | dict[int, int]
):
    def decorator(func):
        sig = inspect.signature(func)

        # Normalize ndim_out into a dict
        if isinstance(ndim_out, int):
            ndim_out_map: dict[int, int] = {0: ndim_out}
        else:
            ndim_out_map = ndim_out

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            outer_shape = None
            unbatched_inputs = set()

            # Flatten or unsqueeze input tensors
            for name, ndim_single in ndim_in.items():
                if name not in bound_args.arguments:
                    continue
                val = bound_args.arguments[name]
                if not isinstance(val, torch.Tensor):
                    raise ValueError(f"Argument '{name}' expected to be a tensor")

                if val.ndim == ndim_single:
                    val = val.unsqueeze(0)
                    unbatched_inputs.add(name)
                elif val.ndim < ndim_single + 1:
                    raise ValueError(f"Tensor '{name}' has too few dimensions")

                current_outer = val.shape if ndim_single == 0 else val.shape[:-ndim_single]
                if outer_shape is None:
                    outer_shape = current_outer
                elif current_outer != outer_shape:
                    raise ValueError(
                        f"Inconsistent outer (batch) shapes across inputs, "
                        f"expected {outer_shape}, got {current_outer}"
                    )

                bound_args.arguments[name] = val.flatten(0, -ndim_single - 1)

            # Call wrapped function
            outputs = func(*bound_args.args, **bound_args.kwargs)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            outputs = list(outputs)
            for i, inner_ndim in ndim_out_map.items():
                out = outputs[i]
                if not isinstance(out, torch.Tensor):
                    raise ValueError(f"Output {i} expected to be a tensor")
                if out.ndim < inner_ndim + 1:
                    raise ValueError(f"Output {i} has too few dimensions")
                outputs[i] = out.unflatten(0, outer_shape)

            if unbatched_inputs:
                for i in ndim_out_map:
                    outputs[i] = outputs[i].squeeze(0)

            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        return wrapper
    return decorator
