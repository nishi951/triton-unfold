import pytest
import torch


@pytest.fixture
def small1d():
    spec = {
        "N": (1,),
        "shape": (15,),
        "block_size": (3,),
        "stride": (1,),
        "mask": torch.tensor([0, 1, 0], dtype=bool),
    }
    return spec


@pytest.fixture
def medium1d():
    spec = {
        "N": (2, 2),
        "shape": (33,),
        "block_size": (7,),
        "stride": (2,),
    }
    return spec


@pytest.fixture
def large1d():
    spec = {
        "N": (20, 1),
        "shape": (1025,),
        "block_size": (25,),
        "stride": (25,),
    }
    return spec


@pytest.fixture
def tiny2d():
    spec = {
        "N": (1,),
        "shape": (4, 4),
        "block_size": (2, 2),
        "stride": (1, 1),
    }
    return spec


@pytest.fixture
def small2d():
    spec = {
        "N": (1,),
        "shape": (15, 15),
        "block_size": (3, 3),
        "stride": (1, 1),
        "mask": torch.tensor(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
    }
    return spec


@pytest.fixture
def medium2d():
    spec = {
        "N": (2, 2),
        "shape": (33, 43),
        "block_size": (7, 7),
        "stride": (2, 2),
    }
    return spec


@pytest.fixture
def large2d():
    spec = {
        "N": (20, 1),
        "shape": (125, 125),
        "block_size": (25, 25),
        "stride": (25, 25),
    }
    return spec


@pytest.fixture
def tiny3d():
    spec = {
        "N": (1,),
        "shape": (4, 4, 4),
        "block_size": (2, 2, 2),
        "stride": (1, 1, 1),
    }
    return spec


@pytest.fixture
def small3d():
    spec = {
        "N": (1,),
        "shape": (15, 15, 15),
        "block_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "mask": torch.tensor(
            [
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
            ],
            dtype=bool,
        ),
    }
    return spec


@pytest.fixture
def medium3d():
    spec = {
        "N": (2, 2),
        "shape": (33, 43, 23),
        "block_size": (7, 7, 5),
        "stride": (2, 2, 2),
    }
    return spec


@pytest.fixture
def large3d():
    spec = {
        "N": (2, 1),
        "shape": (100, 100, 100),
        "block_size": (25, 25, 25),
        "stride": (25, 25, 25),
    }
    return spec
