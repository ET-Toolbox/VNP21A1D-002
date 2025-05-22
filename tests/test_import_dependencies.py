import pytest

# List of dependencies
dependencies = [
    "colored_logging",
    "dateutil",
    "earthaccess",
    "modland",
    "rasters",
    "VIIRS_tiled_granules"
]

# Generate individual test functions for each dependency
@pytest.mark.parametrize("dependency", dependencies)
def test_dependency_import(dependency):
    __import__(dependency)
