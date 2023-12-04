import pytest
import numpy as np

@pytest.fixture
def generate_boxes(n_boxes=100):
    im_size = 10_000
    topleft = np.random.uniform(0.0, high=im_size, size=(n_boxes, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)
    return np.concatenate([topleft, topleft + wh], axis=1).astype(np.float64)
