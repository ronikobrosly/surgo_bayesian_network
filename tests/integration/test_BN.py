""" Integration tests of the network.py module """

import pgmpy
import pytest

from surgo_bayesian_network import Bayes_Net


def test_structure_learning_flow(dataset_fixture):
    """
    Tests the workflow from reading in a dataset to learning a network's structure
    """

    # Save fixture to file
    dataset_fixture.to_csv("tests/integration/test.csv", index=False)

    bn = Bayes_Net("target", verbose=False, random_seed=123)

    bn.read_data("tests/integration/test.csv")
    bn.learn_structure("structure.csv", algorithm="hc")

    assert isinstance(bn.structure_model, pgmpy.base.DAG)
    assert "target" in bn.structure_model.nodes
