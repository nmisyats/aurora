import pytest
import torch
import numpy as np
from aurora.geometry import ray_box_intersection

def test_ray_box_intersection():
    box_min = torch.tensor([-1.0, -2.0, -1.0])
    box_max = torch.tensor([1.0, 3.0, 1.0])

    s10 = float(np.sqrt(10.0))

    cases = torch.tensor([
        [2.0, 0.0, 0.0,    -1.0, 0.0, 0.0,             1.0, 3.0],
        [2.0, 0.0, 0.0,     1.0, 0.0, 0.0,             float('nan'), float('nan')],
        [0.0, 6.0, -1.0,    0.0, -3.0/s10, 1.0/s10,    s10, 2.0*s10],
        [0.5, -1.0, 2.0,    0.0, 0.0, -1.0,            1.0, 3.0],
    ])
    ro, rd = cases[:, 0:3], cases[:, 3:6]
    expected_t_n, expected_t_f = cases[:, 6], cases[:, 7]

    t_n, t_f = ray_box_intersection(ro, rd, box_min, box_max)

    assert torch.allclose(torch.nan_to_num(expected_t_n), torch.nan_to_num(t_n))
    assert torch.allclose(torch.nan_to_num(expected_t_f), torch.nan_to_num(t_f))
