import pytest
import torch
import numpy as np
from aurora.geodesy import az_ze_to_UNE, UNE_to_ECEF, ECEF_to_UNE, UNE_basis_ECEF

# Set tolerance for floating point comparisons
@pytest.fixture
def tol():
    return 1e-5

def test_az_ze_to_UNE_cardinal_directions(tol):
    """Test az_ze_to_UNE with cardinal directions at the horizon."""
    
    # Test cases: (azimuth, zenith, expected_up, expected_north, expected_east)
    cases = torch.tensor([
        [  0.0, 90.0,   0.0,  1.0,  0.0],
        [ 90.0, 90.0,   0.0,  0.0,  1.0],
        [180.0, 90.0,   0.0, -1.0,  0.0],
        [270.0, 90.0,   0.0,  0.0, -1.0],
        [ 45.0, 00.0,   1.0,  0.0,  0.0],
        [-45.0, 00.0,   1.0,  0.0,  0.0],
    ])

    une = az_ze_to_UNE(cases[:,0], cases[:,1])

    assert une.shape == (6,3)
    assert torch.allclose(une, cases[:,2:], atol=tol)

def test_az_ze_to_UNE_unit_vector(tol):
    """Test that az_ze_to_UNE returns unit vectors."""
    
    torch.manual_seed(42)
    num_tests = 50
    azimuth = torch.rand(num_tests) * 360.0
    zenith = torch.rand(num_tests) * 180.0

    une = az_ze_to_UNE(azimuth, zenith)
    magnitude = torch.sqrt(torch.sum(une**2, dim=1))

    assert magnitude.shape == (num_tests,)
    assert torch.allclose(magnitude, torch.ones((num_tests,)), atol=tol)

def test_UNE_to_ECEF_equator_prime_meridian(tol):
    """Test UNE_to_ECEF at the equator and prime meridian (0°N, 0°E)."""
    
    # At equator and prime meridian:
    # - "Up" should point along positive X
    # - "North" should point along positive Z
    # - "East" should point along positive Y
    une = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    expected_ecef = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    ecef = UNE_to_ECEF(une, 0.0, 0.0)

    assert ecef.shape == (3,3)
    assert torch.allclose(ecef, expected_ecef, atol=tol)

def test_UNE_to_ECEF_north_pole(tol):
    """Test UNE_to_ECEF at the North Pole (90°N)."""
    
    # At North Pole:
    # - "Up" should point along positive Z
    # - "North" is undefined, but convention puts it along positive X
    # - "East" is undefined, but convention puts it along positive Y

    up_dir = torch.tensor([1.0, 0.0, 0.0])
    z_dir = torch.tensor([0.0, 0.0, 1.0])
    for lon in [0, 45, 90, 180, 270]:
        ecef = UNE_to_ECEF(up_dir, 90.0, lon)
        torch.allclose(ecef, z_dir, atol=tol)

def test_UNE_ECEF_round_trip(tol):
    """Test that converting UNE to ECEF and back to UNE preserves the original values."""
    
    torch.manual_seed(42)
    num_tests = 10
    
    # Generate random locations and UNE vectors
    lats = torch.rand(num_tests) * 180.0 - 90.0  # -90 to 90
    lons = torch.rand(num_tests) * 360.0 - 180.0  # -180 to 180
    
    azimuths = torch.rand(num_tests) * 360.0  # Random azimuths between 0 and 360
    zeniths = torch.rand(num_tests) * 180.0   # Random zeniths between 0 and 180

    une_original = az_ze_to_UNE(azimuths, zeniths)

    for lat, lon in zip(lats, lons):
        ecef = UNE_to_ECEF(une_original, lat.item(), lon.item())
        une_back = ECEF_to_UNE(ecef, lat.item(), lon.item())
        assert torch.allclose(une_back, une_original, atol=tol)

def test_UNE_basis_ECEF():
    s22 = float(np.sqrt(2.0)/2.0)

    up, north, east = UNE_basis_ECEF(0.0, 0.0)
    assert torch.allclose(up,    torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(north, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(east,  torch.tensor([0.0, 1.0, 0.0]))

    up, north, east = UNE_basis_ECEF(45.0, 0.0)
    assert torch.allclose(up,    torch.tensor([s22,  0.0, s22]))
    assert torch.allclose(north, torch.tensor([-s22, 0.0, s22]))
    assert torch.allclose(east,  torch.tensor([0.0,  1.0, 0.0]))

    up, north, east = UNE_basis_ECEF(-45.0, 90.0)
    assert torch.allclose(up,    torch.tensor([0.0,  s22, -s22]))
    assert torch.allclose(north, torch.tensor([0.0,  s22, s22]))
    assert torch.allclose(east,  torch.tensor([-1.0, 0.0, 0.0]))

if __name__ == "__main__":
    pytest.main(["-v", "-s"])