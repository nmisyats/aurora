import numpy as np
from functools import wraps
import torch

from aurora.data import Camera

def numpify(func=None, *, device="cpu"):
    if func is None:
        return lambda f: numpify(f, device=device)

    @wraps(func)
    def wrapper(*args, **kwargs):
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(device)
            return x

        def to_numpy(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            elif isinstance(x, (list, tuple)):
                return type(x)(to_numpy(i) for i in x)
            elif isinstance(x, dict):
                return {k: to_numpy(v) for k, v in x.items()}
            return x

        args_t = tuple(to_tensor(a) for a in args)
        kwargs_t = {k: to_tensor(v) for k, v in kwargs.items()}
        result = func(*args_t, **kwargs_t)
        return to_numpy(result)

    return wrapper
