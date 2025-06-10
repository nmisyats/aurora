import pytest
import torch
import math
from aurora.geodesy import az_ze_to_enu, rotate_enu_to_ecef, rotate_ecef_to_enu, enu_basis_vectors_ecef

# Set tolerance for floating point comparisons
@pytest.fixture
def tol():
    return 1e-5

def test_az_ze_to_enu_cardinal_directions(tol):
    """Test az_ze_to_ENU with cardinal directions at the horizon."""
    
    # Test cases: (azimuth, zenith, expected_east, expected_north, expected_up)
    cases = torch.tensor([
        [  0.0, 90.0,   0.0,  1.0,  0.0],  # North
        [ 90.0, 90.0,   1.0,  0.0,  0.0],  # East
        [180.0, 90.0,   0.0, -1.0,  0.0],  # South
        [270.0, 90.0,  -1.0,  0.0,  0.0],  # West
        [ 45.0,  0.0,   0.0,  0.0,  1.0],  # Up (zenith)
        [-45.0,  0.0,   0.0,  0.0,  1.0],  # Up (zenith)
    ])

    enu = az_ze_to_enu(cases[:,0], cases[:,1])

    assert enu.shape == (6,3)
    assert torch.allclose(enu, cases[:,2:], atol=tol)

def test_az_ze_to_enu_unit_vector(tol):
    """Test that az_ze_to_ENU returns unit vectors."""
    
    torch.manual_seed(42)
    num_tests = 50
    azimuth = torch.rand(num_tests) * 360.0
    zenith = torch.rand(num_tests) * 180.0

    enu = az_ze_to_enu(azimuth, zenith)
    magnitude = torch.sqrt(torch.sum(enu**2, dim=1))

    assert magnitude.shape == (num_tests,)
    assert torch.allclose(magnitude, torch.ones((num_tests,)), atol=tol)

def test_enu_to_ecef_equator_prime_meridian(tol):
    """Test rotate_ENU_to_ECEF at the equator and prime meridian (0°N, 0°E)."""
    
    # At equator and prime meridian:
    # - "East" should point along positive Y
    # - "North" should point along positive Z
    # - "Up" should point along positive X
    enu = torch.tensor([
        [1.0, 0.0, 0.0],  # East unit vector
        [0.0, 1.0, 0.0],  # North unit vector
        [0.0, 0.0, 1.0],  # Up unit vector
    ])
    expected_ecef = torch.tensor([
        [0.0, 1.0, 0.0],  # East -> Y
        [0.0, 0.0, 1.0],  # North -> Z
        [1.0, 0.0, 0.0],  # Up -> X
    ])
    ecef = rotate_enu_to_ecef(enu, 0.0, 0.0)

    assert ecef.shape == (3,3)
    assert torch.allclose(ecef, expected_ecef, atol=tol)

def test_enu_to_ecef_north_pole(tol):
    """Test rotate_ENU_to_ECEF at the North Pole (90°N)."""
    
    # At North Pole:
    # - "Up" should point along positive Z
    # - "North" is undefined, but convention puts it along positive X
    # - "East" is undefined, but convention puts it along positive Y

    up_dir = torch.tensor([0.0, 0.0, 1.0])  # Up unit vector in ENU
    z_dir = torch.tensor([0.0, 0.0, 1.0])   # Expected Z direction in ECEF
    for lon in [0, 45, 90, 180, 270]:
        ecef = rotate_enu_to_ecef(up_dir, 90.0, lon)
        assert torch.allclose(ecef, z_dir, atol=tol)

def test_enu_ecef_round_trip(tol):
    """Test that converting ENU to ECEF and back to ENU preserves the original values."""
    
    torch.manual_seed(42)
    num_tests = 10
    
    # Generate random locations and ENU vectors
    lats = torch.rand(num_tests) * 180.0 - 90.0  # -90 to 90
    lons = torch.rand(num_tests) * 360.0 - 180.0  # -180 to 180
    
    azimuths = torch.rand(num_tests) * 360.0  # Random azimuths between 0 and 360
    zeniths = torch.rand(num_tests) * 180.0   # Random zeniths between 0 and 180

    enu_original = az_ze_to_enu(azimuths, zeniths)

    for lat, lon in zip(lats, lons):
        ecef = rotate_enu_to_ecef(enu_original, lat.item(), lon.item())
        enu_back = rotate_ecef_to_enu(ecef, lat.item(), lon.item())
        assert torch.allclose(enu_back, enu_original, atol=tol)

def test_enu_basis_vectors_ecef(tol):
    """Test ENU_basis_ECEF function returns correct basis vectors."""
    s22 = math.sqrt(2.0) / 2.0

    # Test at equator and prime meridian (0°N, 0°E)
    east, north, up = enu_basis_vectors_ecef(0.0, 0.0)
    assert torch.allclose(east,  torch.tensor([0.0, 1.0, 0.0]), atol=tol)  # East -> Y
    assert torch.allclose(north, torch.tensor([0.0, 0.0, 1.0]), atol=tol)  # North -> Z
    assert torch.allclose(up,    torch.tensor([1.0, 0.0, 0.0]), atol=tol)  # Up -> X

    # Test at 45°N, 0°E
    east, north, up = enu_basis_vectors_ecef(45.0, 0.0)
    assert torch.allclose(east,  torch.tensor([0.0,  1.0, 0.0]), atol=tol)  # East -> Y
    assert torch.allclose(north, torch.tensor([-s22, 0.0, s22]), atol=tol)  # North
    assert torch.allclose(up,    torch.tensor([s22,  0.0, s22]), atol=tol)  # Up

    # Test at -45°N, 90°E
    east, north, up = enu_basis_vectors_ecef(-45.0, 90.0)
    assert torch.allclose(east,  torch.tensor([-1.0, 0.0, 0.0]), atol=tol)  # East -> -X
    assert torch.allclose(north, torch.tensor([0.0,  s22, s22]), atol=tol)  # North
    assert torch.allclose(up,    torch.tensor([0.0,  s22, -s22]), atol=tol) # Up

def test_enu_basis_orthonormal(tol):
    """Test that ENU basis vectors are orthonormal."""
    torch.manual_seed(42)
    num_tests = 10
    
    # Generate random locations
    lats = torch.rand(num_tests) * 180.0 - 90.0  # -90 to 90
    lons = torch.rand(num_tests) * 360.0 - 180.0  # -180 to 180
    
    for lat, lon in zip(lats, lons):
        east, north, up = enu_basis_vectors_ecef(lat.item(), lon.item())
        
        # Check unit vectors
        assert torch.allclose(torch.norm(east), torch.tensor(1.0), atol=tol)
        assert torch.allclose(torch.norm(north), torch.tensor(1.0), atol=tol)
        assert torch.allclose(torch.norm(up), torch.tensor(1.0), atol=tol)
        
        # Check orthogonality
        assert torch.allclose(torch.dot(east, north), torch.tensor(0.0), atol=tol)
        assert torch.allclose(torch.dot(east, up), torch.tensor(0.0), atol=tol)
        assert torch.allclose(torch.dot(north, up), torch.tensor(0.0), atol=tol)

if __name__ == "__main__":
    pytest.main(["-v", "-s"])