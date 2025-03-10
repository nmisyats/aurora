import pytest
import torch
from aurora.geodesy import az_ze_to_UNE, UNE_to_ECEF, ECEF_to_UNE

# Set tolerance for floating point comparisons
@pytest.fixture
def tol():
    return 1e-5

# For simpler notations
@pytest.fixture
def T():
    return lambda x: torch.tensor([x])

def test_az_ze_to_UNE_cardinal_directions(tol, T):
    """Test az_ze_to_UNE with cardinal directions at the horizon."""
    
    # Test cases: (azimuth, zenith, expected_up, expected_north, expected_east)
    test_cases = [
        # North direction (azimuth=0°, zenith=90°)
        (T(0.0), T(90.0), 
         T(0.0), T(1.0), T(0.0)),
        
        # East direction (azimuth=90°, zenith=90°)
        (T(90.0), T(90.0), 
         T(0.0), T(0.0), T(1.0)),
        
        # South direction (azimuth=180°, zenith=90°)
        (T(180.0), T(90.0), 
         T(0.0), T(-1.0), T(0.0)),
        
        # West direction (azimuth=270°, zenith=90°)
        (T(270.0), T(90.0), 
         T(0.0), T(0.0), T(-1.0)),
        
        # Directly up (azimuth=any, zenith=0°)
        (T(45.0), T(0.0), 
         T(1.0), T(0.0), T(0.0)),
    ]
    
    for azimuth, zenith, exp_up, exp_north, exp_east in test_cases:
        up, north, east = az_ze_to_UNE(azimuth, zenith)
        
        assert torch.isclose(up, exp_up, atol=tol), f"Up component failed for azimuth={azimuth.item()}, zenith={zenith.item()}"
        assert torch.isclose(north, exp_north, atol=tol), f"North component failed for azimuth={azimuth.item()}, zenith={zenith.item()}"
        assert torch.isclose(east, exp_east, atol=tol), f"East component failed for azimuth={azimuth.item()}, zenith={zenith.item()}"

def test_az_ze_to_UNE_unit_vector(tol):
    """Test that az_ze_to_UNE returns unit vectors."""
    
    # Generate random azimuth and zenith angles
    torch.manual_seed(42)  # For reproducibility
    num_tests = 50
    azimuth = torch.rand(num_tests) * 360.0  # Random azimuths between 0 and 360
    zenith = torch.rand(num_tests) * 180.0   # Random zeniths between 0 and 180

    up, north, east = az_ze_to_UNE(azimuth, zenith)
        
    # Calculate magnitude of the resulting vector
    magnitude = torch.sqrt(up**2 + north**2 + east**2)
    
    # Check that each are a unit vector
    for i in range(num_tests):
        assert torch.isclose(magnitude[i], torch.tensor([1.0]), atol=tol), \
            f"Not a unit vector for azimuth={azimuth[i].item()}, zenith={zenith[i].item()}"

def test_UNE_to_ECEF_equator_prime_meridian(tol, T):
    """Test UNE_to_ECEF at the equator and prime meridian (0°N, 0°E)."""
    
    # At equator and prime meridian:
    # - "Up" should point along positive X
    # - "North" should point along positive Z
    # - "East" should point along positive Y
    lat = T(0.0)
    lon = T(0.0)
    
    # Test Up direction
    une = (T(1.0), T(0.0), T(0.0))
    x, y, z = UNE_to_ECEF(une, lat, lon)
    assert torch.isclose(x, T(1.0), atol=tol)
    assert torch.isclose(y, T(0.0), atol=tol)
    assert torch.isclose(z, T(0.0), atol=tol)
    
    # Test North direction
    une = (T(0.0), T(1.0), T(0.0))
    x, y, z = UNE_to_ECEF(une, lat, lon)
    assert torch.isclose(x, T(0.0), atol=tol)
    assert torch.isclose(y, T(0.0), atol=tol)
    assert torch.isclose(z, T(1.0), atol=tol)
    
    # Test East direction
    une = (T(0.0), T(0.0), T(1.0))
    x, y, z = UNE_to_ECEF(une, lat, lon)
    assert torch.isclose(x, T(0.0), atol=tol)
    assert torch.isclose(y, T(1.0), atol=tol)
    assert torch.isclose(z, T(0.0), atol=tol)

def test_UNE_to_ECEF_north_pole(tol, T):
    """Test UNE_to_ECEF at the North Pole (90°N)."""
    
    # At North Pole:
    # - "Up" should point along positive Z
    # - "North" is undefined, but convention puts it along positive X
    # - "East" is undefined, but convention puts it along positive Y
    lat = T(90.0)
    lon = T(0.0)  # Longitude doesn't matter at poles
    
    # Test Up direction
    une = (T(1.0), T(0.0), T(0.0))
    x, y, z = UNE_to_ECEF(une, lat, lon)
    assert torch.isclose(x, T(0.0), atol=tol)
    assert torch.isclose(y, T(0.0), atol=tol)
    assert torch.isclose(z, T(1.0), atol=tol)

def test_UNE_ECEF_round_trip(tol):
    """Test that converting UNE to ECEF and back to UNE preserves the original values."""
    
    torch.manual_seed(42)  # For reproducibility
    num_tests = 10
    
    # Generate random locations and UNE vectors
    lats = torch.rand(num_tests) * 180.0 - 90.0  # -90 to 90
    lons = torch.rand(num_tests) * 360.0 - 180.0  # -180 to 180
    
    azimuths = torch.rand(num_tests) * 360.0  # Random azimuths between 0 and 360
    zeniths = torch.rand(num_tests) * 180.0   # Random zeniths between 0 and 180
    
    ups, norths, easts = az_ze_to_UNE(azimuths, zeniths)
    
    for i in range(num_tests):
        lat = lats[i:i+1]
        lon = lons[i:i+1]
        
        # Create UNE vector
        une_original = (
            ups[i:i+1],
            norths[i:i+1],
            easts[i:i+1]
        )
        
        # Forward conversion: UNE to ECEF
        ecef = UNE_to_ECEF(une_original, lat, lon)
        
        # Backward conversion: ECEF to UNE
        une_back = ECEF_to_UNE(ecef, lat, lon)
        
        # Check that original and round-trip values match
        assert torch.isclose(une_original[0], une_back[0], atol=tol), \
            f"Up component mismatch: original={une_original[0].item()}, round-trip={une_back[0].item()}"
        assert torch.isclose(une_original[1], une_back[1], atol=tol), \
            f"North component mismatch: original={une_original[1].item()}, round-trip={une_back[1].item()}"
        assert torch.isclose(une_original[2], une_back[2], atol=tol), \
            f"East component mismatch: original={une_original[2].item()}, round-trip={une_back[2].item()}"

def test_UNE_ECEF_specific_locations(tol, T):
    """Test UNE to ECEF to UNE round-trip conversion at specific locations."""
    
    # Test cases: (up, north, east, lat, lon)
    test_cases = [
        # At equator, prime meridian
        (T(1.0), T(0.0), T(0.0), T(0.0), T(0.0)),
        (T(0.0), T(1.0), T(0.0), T(0.0), T(0.0)),
        (T(0.0), T(0.0), T(1.0), T(0.0), T(0.0)),
        
        # At North Pole
        (T(1.0), T(0.0), T(0.0), T(90.0), T(0.0)),
        
        # At 45°N, 45°E
        (T(1.0), T(0.0), T(0.0), T(45.0), T(45.0)),
        (T(0.0), T(1.0), T(0.0), T(45.0), T(45.0)),
        (T(0.0), T(0.0), T(1.0), T(45.0), T(45.0)),
    ]
    
    for up, north, east, lat, lon in test_cases:
        # Create UNE vector
        une_original = (up, north, east)
        
        # Forward conversion: UNE to ECEF
        ecef = UNE_to_ECEF(une_original, lat, lon)
        
        # Backward conversion: ECEF to UNE
        une_back = ECEF_to_UNE(ecef, lat, lon)
        
        # Check that original and round-trip values match
        assert torch.isclose(une_original[0], une_back[0], atol=tol), \
            f"Up component mismatch at lat={lat.item()}, lon={lon.item()}: original={une_original[0].item()}, round-trip={une_back[0].item()}"
        assert torch.isclose(une_original[1], une_back[1], atol=tol), \
            f"North component mismatch at lat={lat.item()}, lon={lon.item()}: original={une_original[1].item()}, round-trip={une_back[1].item()}"
        assert torch.isclose(une_original[2], une_back[2], atol=tol), \
            f"East component mismatch at lat={lat.item()}, lon={lon.item()}: original={une_original[2].item()}, round-trip={une_back[2].item()}"

if __name__ == "__main__":
    pytest.main(["-v"])