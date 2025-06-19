import pytest
import torch
from aurora.utils import normalize_batch_dims

@pytest.fixture(scope="module")
def foo_1d():
    @normalize_batch_dims({"x": 1}, {0: 1})
    def func(x: torch.Tensor, param: torch.Tensor, label: str):
        return x + param, label
    return func

@pytest.fixture(scope="module")
def foo_2d():
    @normalize_batch_dims({"x": 1}, {0: 1})
    def func(x: torch.Tensor, param: torch.Tensor, label: str):
        return x + param.unsqueeze(0), label
    return func

def test_normalize_batch_dims_single(foo_1d):
    x = torch.tensor([1.0, 2.0])
    param = torch.tensor([10.0, 20.0])
    result, label = foo_1d(x, param, "test")
    assert torch.allclose(result, torch.tensor([11.0, 22.0]))
    assert label == "test"

def test_normalize_batch_dims_flat_batch_1d_in(foo_1d):
    x = torch.tensor(
        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]])
    param = torch.tensor([10.0, 20.0])
    result, label = foo_1d(x, param, "test")
    assert torch.allclose(result, torch.tensor(
        [[11.0, 22.0],
         [13.0, 24.0],
         [15.0, 26.0]]))
    assert label == "test"

def test_normalize_batch_dims_flat_batch_2d_in(foo_2d):
    x = torch.tensor(
        [[[1.0, 2.0],
          [3.0, 4.0],
          [5.0, 6.0]],
        
         [[6.0, 5.0],
          [4.0, 3.0],
          [2.0, 1.0]]])
    param = torch.tensor([10.0, 20.0])
    result, label = foo_2d(x, param, "test")
    assert torch.allclose(result, torch.tensor(
       [[[11.0, 22.0],
         [13.0, 24.0],
         [15.0, 26.0]],
        
        [[16.0, 25.0],
         [14.0, 23.0],
         [12.0, 21.0]]]))
    assert label == "test"

def test_normalize_batch_dims_2d_batch_1d_in(foo_1d):
    x = torch.tensor(
        [[[1.0, 2.0],
          [3.0, 4.0],
          [5.0, 6.0]],
        
         [[6.0, 5.0],
          [4.0, 3.0],
          [2.0, 1.0]]])
    param = torch.tensor([10.0, 20.0])
    result, label = foo_1d(x, param, "test")
    assert torch.allclose(result, torch.tensor(
       [[[11.0, 22.0],
         [13.0, 24.0],
         [15.0, 26.0]],
        
        [[16.0, 25.0],
         [14.0, 23.0],
         [12.0, 21.0]]]))
    assert label == "test"

def test_normalize_batch_dims_2d_batch_2d_in(foo_1d):
    x = torch.tensor(
        [[[[1.0, 2.0],
           [3.0, 4.0],
           [5.0, 6.0]],
        
          [[6.0, 5.0],
           [4.0, 3.0],
           [2.0, 1.0]]],
         
         [[[6.0, 5.0],
           [4.0, 3.0],
           [2.0, 1.0]],
        
          [[1.0, 2.0],
           [3.0, 4.0],
           [5.0, 6.0]]]])
    param = torch.tensor([10.0, 20.0])
    result, label = foo_1d(x, param, "test")
    assert torch.allclose(result, torch.tensor(
        [[[[11.0, 22.0],
           [13.0, 24.0],
           [15.0, 26.0]],
        
          [[16.0, 25.0],
           [14.0, 23.0],
           [12.0, 21.0]]],
         
         [[[16.0, 25.0],
           [14.0, 23.0],
           [12.0, 21.0]],
        
          [[11.0, 22.0],
           [13.0, 24.0],
           [15.0, 26.0]]]]))
    assert label == "test"
