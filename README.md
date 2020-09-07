# Surgo Bayesian Network

A python package case study for Surgo Foundation



## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Tests](#tests)


## Overview

This package contains a tool that allows an entry-level python user to learn the
structure of a Bayesian Network from data, visualize the network, and produces a
summary of the influence of predictors on a specified target variable in the network.


## Installation

This isn't publicly-available on PyPI for obvious reasons. Feel free to clone locally via:

`git clone -b master https://github.com/ronikobrosly/surgo_bayesian_network.git`

Create a virtual environment, navigate to the project folder and run:

`pip install -r requirements.txt`

## Requirements

This tool requires python >= 3.7.6 and pgmpy 0.1.11


## Quickstart

Within your virtual environment with all dependencies installed, navigate to the root folder
of the `surgo_bayesian_network` project, import the `Bayes_Net` class, and you can begin using the tool:

```
from surgo_bayesian_network import Bayes_Net

bn = Bayes_Net(target_variable = "B", verbose = True, random_seed = 111)
bn.read_data("~/Desktop/surgo/data_5000samples.csv")
bn.learn_structure("~/Desktop/structure.csv", algorithm = 'hc')
bn.plot_network("~/Desktop/structure.png")
bn.plot_causal_influence("~/Desktop/forest_plot.png")

```

## Tests

To run the example unit and integration tests, install the python package `pytest` and within the
root folder of the `surgo_bayesian_network` project run the following command:

```pytest tests/```
