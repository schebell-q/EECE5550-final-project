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


def make_long_toy_env() -> Environment:
    points = np.array([
        [0, 0],  # 0
        [0, 17],  # 1
        [4, 1],  # 2
        [4, 2],  # 3
        [-1, 2],  # 4
        [-1, 1],  # 5
        [1, 3],  # 6
        [1, 4],  # 7
        [-4, 4],  # 8
        [-4, 3],  # 9
        [6, 5],  # 10
        [6, 6],  # 11
        [-1, 6],  # 12
        [-1, 5],  # 13
        [1, 7],  # 14
        [1, 8],  # 15
        [-8, 8],  # 16
        [-8, 7],  # 17
        [8, 9],  # 18
        [8, 10],  # 19
        [-1, 10],  # 20
        [-1, 9],  # 21
        [1, 11],  # 22
        [1, 12],  # 23
        [-6, 12],  # 24
        [-6, 11],  # 25
        [4, 13],  # 26
        [4, 14],  # 27
        [-1, 14],  # 28
        [-1, 13],  # 29
        [1, 15],  # 30
        [1, 16],  # 31
        [-4, 16],  # 32
        [-4, 15],  # 33
    ], dtype="float64")
    points[:, 1] /= 2
    low_traversability = 0.2
    high_traversability = 0.8
    regions = {
        Region((2, 3, 4, 5), high_traversability),
        Region((6, 7, 8, 9), high_traversability),
        Region((10, 11, 12, 13), high_traversability),
        Region((14, 15, 16, 17), high_traversability),
        Region((18, 19, 20, 21), low_traversability),
        Region((22, 23, 24, 25), low_traversability),
        Region((26, 27, 28, 29), low_traversability),
        Region((30, 31, 32, 33), low_traversability),
    }
    return Environment(0, 1, points, regions)


def test_distance_between():
    env = make_basic_env()
    assert env.distance_between(0, 1) == 3
    assert env.distance_between(2, 3) == 1
    assert env.distance_between(4, 1) == np.sqrt(2)


def test_add_redundant_points():
    env = make_basic_env()
    assert env._n_points == 6
    assert env.add_point((0, 0.00001)) == 0
    assert env.add_point((2.999999, -0.00001)) == 1
    assert env._n_points == 6
    assert env.add_point((1, 0)) == 6
    assert env._n_points == 7
