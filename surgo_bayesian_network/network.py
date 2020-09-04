"""
Bayesian network module
"""

import pandas as pd
from pgmpy.estimators import BicScore, ConstraintBasedEstimator, HillClimbSearch
from pgmpy.independencies import Independencies

from surgo_bayesian_network.core import Core


class Bayes_Net(Core):
    """
    Methods to read_in data and learn the structure and conditional probability tables for
    a Bayesian Network, as well as assessing the strength of the causal influence of endogenous
    variables on the target variable of interest.

    Parameters
    ----------
    verbose: bool, optional (default = False)
        Determines whether the user will get verbose status updates.
    Attributes
    ----------
    grid_values: array of shape (treatment_grid_num, )
        The gridded values of the treatment variable. Equally spaced.
    gps_deviance: float
        The GPS model deviance
    Methods
    ----------
    read_data: (self, T, X, y)
        Reads in dataset using. Essentially a wrapper for pandas' `read_csv` function.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

        # Validate the params
        self._validate_init_params()

        if self.verbose:
            print("Using the following params for GPS model:")
            pprint(self.get_params(), indent=4)



    def _validate_init_params(self):
        """
        Checks that the params used when instantiating Bayes_Net are formatted correctly
        """
        # Checks for verbose
        if not isinstance(self.verbose, bool):
            raise TypeError(
                f"verbose parameter must be a boolean type, but found type {type(self.verbose)}"
            )



    def read_data(self, file_path, target_variable, **kwargs):
        """
        Wrapper for pandas `read_csv` function. Assumes file is CSV with a header row.

        Arguments:
            file_path: str, the absolute file path to the CSV file
            target_variable: str, name of the column containing the outcome variable.
            **kwargs: any additional keywords for pandas' `read_csv` function

        Returns:
            None
        """
        self.df = pd.read_csv(filepath_or_buffer = file_path, **kwargs)
        self.target = target_variable
        if target_variable not in self.df:
            raise ValueError("The target variable you specified isn't in the dataset!")
        if self.verbose:
            print("Successfully read in CSV")
        return None


    def learn_structure(self, algorithm = 'hc'):
        """
        Employs `pgmpy` package's Bayesian Network structure learning algorithms to learn
        structure from a dataset

        Arguments:
            algorithm: str, optional (default = 'hc')
                Determines whether the hill-climbing or Peter-Clark are employed.
                Two possible values include: 'hc', 'pc'.

        Returns:
            None
        """
        self.structure_algorithm = algorithm

        if self.verbose:
            print("Depending on the number of variables in your dataset, this might take some time...")

        if algorithm == "hc":
            self.structure_model = HillClimbSearch(bn.df, scoring_method=BicScore(bn.df)).estimate()
        else:
            pass




        if self.verbose:
            print("Structure learned!")









########################################
########################################
########################################



bn = Bayes_Net()
bn.read_data("~/Desktop/surgo/data_5000samples.csv", "B")



hc = HillClimbSearch(bn.df, scoring_method=BicScore(bn.df))
best_model = hc.estimate()
print(best_model.edges())




ind = Independencies()
pc = ConstraintBasedEstimator(bn.df)
model = pc.estimate(significance_level = 0.5)
print(model.edges())



# import itertools
# pairs = list(itertools.product(bn.df.columns, repeat=2))

# skel, sep_sets = pc.build_skeleton(nodes = bn.df.columns, independencies = ind)
# print(skel.edges())
# model = DAG([('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E')])
