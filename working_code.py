
from os.path import expanduser
import pdb
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, BicScore, ConstraintBasedEstimator, HillClimbSearch
from pgmpy.independencies import Independencies
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianModel
from pylab import rcParams
import scipy.stats as ss

from surgo_bayesian_network import Bayes_Net



bn = Bayes_Net(target_variable = "B", verbose = True, random_seed = 111)
bn.read_data("~/Desktop/surgo/data_5000samples.csv")
bn.learn_structure("~/Desktop/structure.csv", algorithm = 'hc')
bn.plot_network("~/Desktop/structure.png")
bn.plot_causal_influence("~/Desktop/forest_plot.png")
