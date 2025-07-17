import pytest
import torch
import math
from aurora.geometry import ray_box_intersection

def test_ray_box_intersection():
    box_min = torch.tensor([-1.0, -2.0, -1.0])
    box_max = torch.tensor([1.0, 3.0, 1.0])

    s10 = math.sqrt(10.0)

    cases = torch.tensor([
        [2.0, 0.0, 0.0,    -1.0, 0.0, 0.0,             1.0, 3.0],
        [2.0, 0.0, 0.0,     1.0, 0.0, 0.0,             torch.nan, torch.nan],
        [0.0, 6.0, -1.0,    0.0, -3.0/s10, 1.0/s10,    s10, 2.0*s10],
        [0.5, -1.0, 2.0,    0.0, 0.0, -1.0,            1.0, 3.0],
    ])
    ro, rd = cases[:, 0:3], cases[:, 3:6]
    expected_tn, expected_tf = cases[:, 6], cases[:, 7]

    tn, tf = ray_box_intersection(ro, rd, box_min, box_max)

    assert torch.allclose(torch.nan_to_num(expected_tn), torch.nan_to_num(tn))
    assert torch.allclose(torch.nan_to_num(expected_tf), torch.nan_to_num(tf))

def test_ray_box_intersection_infty():
    box_min = torch.tensor([-torch.inf, -torch.inf, -1.0])
    box_max = torch.tensor([ torch.inf,  torch.inf,  1.0])

    cases = torch.tensor([
        [0.0,  0.0, -2.0,    0.0, 0.0,  1.0,    1.0, 3.0],
        [5.0, -7.0,  3.0,    0.0, 0.0, -1.0,    2.0, 4.0],
        [5.0, -7.0,  3.0,    0.0, 0.0,  1.0,    torch.nan, torch.nan],
        [5.0, -7.0,  3.0,    1.0, 0.0,  0.0,    torch.nan, torch.nan],
    ])
    ro, rd = cases[:, 0:3], cases[:, 3:6]
    expected_tn, expected_tf = cases[:, 6], cases[:, 7]

    tn, tf = ray_box_intersection(ro, rd, box_min, box_max)

    assert torch.allclose(torch.nan_to_num(expected_tn), torch.nan_to_num(tn))
    assert torch.allclose(torch.nan_to_num(expected_tf), torch.nan_to_num(tf))
