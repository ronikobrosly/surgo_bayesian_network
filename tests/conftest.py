"""Common fixtures for tests using pytest framework"""


import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def dataset_fixture():
    """Returns full_example_dataset"""
    return full_example_dataset()


def full_example_dataset():
    """Example dataset with a target variable and a few `intervention` variables"""

    np.random.seed(500)

    target = np.random.randint(low=0, high=2, size=100)
    x1 = target.copy()
    x2 = target.copy()

    x1[2] = 2
    x1[9:12] = 2
    x1[15:20] = 0
    x1[33] = 2
    x1[38:44] = 0
    x2[4:7] = 1
    x2[9:12] = 1
    x2[15:17] = 1
    x2[23:26] = 1
    x2[33:44] = 0

    fixture = pd.DataFrame({"target": target, "x1": x1, "x2": x2})

    fixture.reset_index(drop=True, inplace=True)

    return fixture
