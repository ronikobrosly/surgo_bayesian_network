"""
Bayesian network module
"""

from os.path import expanduser

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, BicScore, ConstraintBasedEstimator, HillClimbSearch
from pgmpy.independencies import Independencies
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianModel

from surgo_bayesian_network.core import Core



class Bayes_Net(Core):
    """
    Methods to read_in data and learn the structure and conditional probability tables for
    a Bayesian Network, as well as assessing the strength of the causal influence of endogenous
    variables on the target variable of interest.


    Parameters
    ----------
    target_variable: str, name of the column containing the outcome variable.

    verbose: bool, optional (default = False). Determines whether the user will get verbose status updates.

    random_seed: int, optional.


    Attributes
    ----------
    verbose: boolean
        Whether verbose mode is activated

    target_variable: string
        Name of the target variable in the dataset

    df: pd.DataFrame
        pandas dataframe of input dataset

    structure_algorithm: string
        Name of the learning structure algo that was chosen

    structure_model: pgmpy.base.DAG.DAG
        Learned DAG but without conditional probability tables estimated

    bn_model: pgmpy.models.BayesianModel
        Proper, learned Bayesian Network with conditional probability tables estimated


    Methods
    ----------
    read_data: (self, file_path, **kwargs)
        Reads in dataset using. Essentially a wrapper for pandas' `read_csv` function.

    learn_structure: (self, file_path, algorithm = 'hc')
        Learns the structure of a DAG from data. Saves structure as a CSV to disk.
        Note: this is technically not a bayesian network yet, as we don't have the
        conditional probability tables estimated yet.

    plot_network: (self, file_path, **kwargs)
        Plots the Bayesian Network (highlighting target variable) and saves PNG to disk.

    estimate_CPT: (self)
        Estimates the conditional probability tables for each variable / node in the network.

    plot_causal_influence: (self, file_path)
        Uses belief propagation to perform inference and calculates odds ratios for how
        changes in intervention evidence will impact the target variable. A forest plot is
        produced from this.
    """

    def __init__(self, target_variable, random_seed, verbose=False):
        self.verbose = verbose
        self.target_variable = target_variable
        self.random_seed = random_seed


        # Validate the params
        self._validate_init_params()

        if self.verbose:
            print("Using the following params for GPS model:")
            pprint(self.get_params(), indent=4)



    def _validate_init_params(self):
        """
        Very basic checks that the params used when instantiating Bayes_Net look okay
        """
        # Checks for target_variable
        if not isinstance(self.target_variable, str):
            raise TypeError(
                f"target_variable parameter must be a string type, but found type {type(self.target_variable)}"
            )

        # Checks for verbose
        if not isinstance(self.verbose, bool):
            raise TypeError(
                f"verbose parameter must be a boolean type, but found type {type(self.verbose)}"
            )

    def read_data(self, file_path, **kwargs):
        """
        Wrapper for pandas `read_csv` function. Assumes file is CSV with a header row.

        Arguments:
            file_path: str, the absolute file path to the CSV file
            **kwargs: any additional keywords for pandas' `read_csv` function

        Returns:
            None
        """
        self.df = pd.read_csv(filepath_or_buffer = file_path, **kwargs)

        # Check that target variable is in the dataset
        if self.target_variable not in self.df:
            raise ValueError("The target variable you specified isn't in the dataset!")

        if self.verbose:
            print("Successfully read in CSV")

        return None


    def learn_structure(self, file_path, algorithm = 'hc'):
        """
        Employs `pgmpy` package's Bayesian Network structure learning algorithms to learn
        structure from a dataset. Saves a tabular version of the result as a CSV file.

        Arguments:
            algorithm: str, optional (default = 'hc')
                Determines whether the hill-climbing or Peter-Clark are employed.
                Two possible values include: 'hc', 'pc'. Note, I found a bug in pgmpy implementation
                halfway through this project. Don't use the 'pc' method.
            file_path: str, the absolute path to save the file to (e.g. "~/Desktop/BN_structure.csv")

        Returns:
            None
        """
        self.structure_algorithm = algorithm

        if self.verbose:
            print("Depending on the number of variables in your dataset, this might take some time...")

        # Learn structure, using one of the algorithms
        np.random.seed(self.random_seed)

        if algorithm == "hc":
            self.structure_model = HillClimbSearch(bn.df, scoring_method=BicScore(bn.df)).estimate()
        elif algorithm == "pc":
            self.structure_model = ConstraintBasedEstimator(bn.df).estimate(significance_level = 0.10)

        if self.verbose:
            print(f"Structure learned! Saving structure to the following CSV: {file_path}")

        pd.DataFrame(list(self.structure_model.edges), columns =["from_variable", "to_variable"]).to_csv("~/Desktop/BN_structure.csv", index = False)


    def plot_network(self, file_path, **kwargs):
        """
        Plots the learned structure, highlighting the target variable.

        Arguments:
            file_path: str, the absolute path to save the file to (e.g. "~/Desktop/plot.png")
            **kwargs: additional keyword arguments for networkx's draw function

        Returns:
            None
        """
        if self.verbose:
            print(f"Saving Bayesian Network plot to the following PNG file: {file_path}")

        # Identify target variable so we can highlight it in the plot
        target_index = list(self.structure_model).index(self.target_variable)
        node_size_list = [300] * len(list(self.structure_model.nodes))
        node_color_list = ['#95ABDF'] * len(list(self.structure_model.nodes))
        node_size_list[target_index] = 1500
        node_color_list[target_index] = '#F09A9A'

        # Clear any existing pyplot fig, create plot, and save to disk
        plt.clf()
        nx.draw(self.structure_model, node_size = node_size_list, node_color = node_color_list, with_labels=True, **kwargs)
        plt.savefig(expanduser(file_path), format="PNG", dpi = 300)


    def estimate_CPT(self):
        """
        Estimates the conditional probability tables associated with each node in the
        Bayesian Network.

        Arguments:
            None

        Returns:
            None
        """
        self.bn_model = BayesianEstimator(BayesianModel(list(bn.structure_model.edges)), bn.df)


    def plot_causal_influence(self, file_path):
        """
        Computes the odds of the target variable being value 1 over value 0 (i.e. the odds ratio)
        by iterating through all other network variables/nodes, changing their values,
        and observing how the probability of the target variable changes. Belief propagation
        is used for inference. A forest plot is produced from this and saved to disk.

        Arguments:
            file_path: str, the absolute path to save the file to (e.g. "~/Desktop/forest_plot.png")

        Returns:
            None
        """

        if self.verbose:
            print(f"Saving Bayesian Network plot to the following PNG file: {file_path}")

        if self.verbose:
            print(f"Saving forest plot to the following PNG file: {file_path}")

        pass
