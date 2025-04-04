import numpy as np

from src.geometry import Environment, Region


def make_basic_env() -> Environment:
    points = np.array([
        [0, 0],   # 0
        [3, 0],   # 1
        [1, 1],   # 2
        [2, 1],   # 3
        [2, -1],  # 4
        [1, -1],  # 5
    ])
    regions = {
        Region((2, 3, 4, 5), 0.5)
    }
    return Environment(0, 1, points, regions)


def test_distance_between():
    env = make_basic_env()
    assert env.distance_between(0, 1) == 3
    assert env.distance_between(2, 3) == 1
    assert env.distance_between(4, 1) == np.sqrt(2)
