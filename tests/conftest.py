import pytest

from homozip.model import Params


@pytest.fixture
def prm() -> Params:
    """The default calibration, with a short simulation."""
    return Params(n_cells=50, n_points=201, t_end=200.0)
