""" Unit tests of the network.py module """

import pytest

from surgo_bayesian_network import Bayes_Net


def test_cramers_v(dataset_fixture):
    """
    Tests the implementation of Cramer's V correlation calculation
    """

    bn = Bayes_Net("target")

    observed_result = round(
        bn._cramers_v(dataset_fixture["target"], dataset_fixture["x1"]), 4
    )
    expected_result = 0.8594

    assert isinstance(observed_result, float)
    assert observed_result == expected_result
