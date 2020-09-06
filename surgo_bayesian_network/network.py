"""
Bayesian network module
"""

from os.path import expanduser
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import (
    BayesianEstimator,
    BicScore,
    ConstraintBasedEstimator,
    HillClimbSearch,
)
from pgmpy.independencies import Independencies
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianModel
from pylab import rcParams
import scipy.stats as ss

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

    odds_ratios: pd.DataFrame
        DataFrame containing odds ratios for all interventions and levels


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

    plot_causal_influence: (self, file_path)
        Uses belief propagation to perform inference and calculates odds ratios for how
        changes in intervention evidence will impact the target variable. A forest plot is
        produced from this.
    """

    def __init__(self, target_variable, random_seed=0, verbose=False):
        self.verbose = verbose
        self.target_variable = target_variable
        self.random_seed = random_seed

        # Validate the params
        self._validate_init_params()

        if self.verbose:
            print("Using the following params for Bayesian Network model:")
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

        # Checks for random_seed
        if not isinstance(self.random_seed, (int, type(None))):
            raise TypeError(
                f"random_seed parameter must be an int, but found type {type(self.random_seed)}"
            )

        if (isinstance(self.random_seed, int)) and self.random_seed < 0:
            raise ValueError(f"random_seed parameter must be > 0")

    def read_data(self, file_path, **kwargs):
        """
        Wrapper for pandas `read_csv` function. Assumes file is CSV with a header row.

        Arguments:
            file_path: str, the absolute file path to the CSV file
            **kwargs: any additional keywords for pandas' `read_csv` function

        Returns:
            None
        """
        self.df = pd.read_csv(filepath_or_buffer=file_path, **kwargs)

        # Check that target variable is in the dataset
        if self.target_variable not in self.df:
            raise ValueError("The target variable you specified isn't in the dataset!")

        if self.verbose:
            print("Successfully read in CSV")

        return None

    def _cramers_v(self, x, y):
        """
        Static method to that calculates Cramers V correlation between two categorical variables
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]

        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))

        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)

        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    def _initial_filter(self):
        """
        Filters out nodes with zero correlation with target variable
        """

        relevant_vars = []

        for node in self.df.columns:
            if self._cramers_v(self.df[self.target_variable], self.df[node]) > 0:
                relevant_vars.append(node)

        return self.df[relevant_vars]

    def learn_structure(self, file_path, algorithm="hc", significance_level=0.10):
        """
        Employs `pgmpy` package's Bayesian Network structure learning algorithms to learn
        structure from a dataset. Saves a tabular version of the result as a CSV file.

        Arguments:
            algorithm: str, optional (default = 'hc')
                Determines whether the hill-climbing or Peter-Clark are employed.
                Two possible values include: 'hc', 'pc'. Note, I found a bug in pgmpy implementation
                halfway through this project. Don't use the 'pc' method.
            file_path: str, the absolute path to save the file to (e.g. "~/Desktop/BN_structure.csv")
            significance_level: float, option (default = 0.10)
                Statistical significance cutoff for use in pruning the network when using the PC
                algorithm. Lower values produce sparser networks.

        Returns:
            None
        """
        self.structure_algorithm = algorithm

        if self.verbose:
            print(
                "Depending on the number of variables in your dataset, this might take some time..."
            )

        # Learn structure, using one of the algorithms
        np.random.seed(self.random_seed)

        if algorithm == "hc":

            # Filter out columns with zero correlation with target variable
            self.filtered_df = self._initial_filter()

            # Run HC algorithm
            self.structure_model = HillClimbSearch(
                self.filtered_df, scoring_method=BicScore(self.filtered_df)
            ).estimate()

            if self.verbose:
                print(
                    f"Structure learned! Saving structure to the following CSV: {file_path}"
                )

            # Eliminate isolated subgraphs
            G = self.structure_model.to_undirected()

            connected_nodes = list(
                nx.algorithms.components.node_connected_component(
                    G, self.target_variable
                )
            )

            disconnected_nodes = list(
                set(list(self.structure_model.nodes)) - set(connected_nodes)
            )

            for node in disconnected_nodes:
                self.structure_model.remove_node(node)
                self.filtered_df.drop([node], axis=1, inplace=True)

            pd.DataFrame(
                list(self.structure_model.edges),
                columns=["from_variable", "to_variable"],
            ).to_csv("~/Desktop/BN_structure.csv", index=False)

        elif algorithm == "pc":
            self.filtered_df = self.df
            self.structure_model = ConstraintBasedEstimator(self.filtered_df).estimate(
                significance_level=significance_level
            )

            if self.verbose:
                print(
                    f"Structure learned! Saving structure to the following CSV: {file_path}"
                )

            pd.DataFrame(
                list(self.structure_model.edges),
                columns=["from_variable", "to_variable"],
            ).to_csv("~/Desktop/BN_structure.csv", index=False)

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
            print(
                f"Saving Bayesian Network plot to the following PNG file: {file_path}"
            )

        # Identify target variable so we can highlight it in the plot
        target_index = list(self.structure_model).index(self.target_variable)
        node_size_list = [300] * len(list(self.structure_model.nodes))
        node_color_list = ["#95ABDF"] * len(list(self.structure_model.nodes))
        node_size_list[target_index] = 1500
        node_color_list[target_index] = "#F09A9A"

        # Clear any existing pyplot fig, create plot, and save to disk
        plt.clf()
        nx.draw(
            self.structure_model,
            node_size=node_size_list,
            node_color=node_color_list,
            with_labels=True,
            **kwargs,
        )
        plt.savefig(expanduser(file_path), format="PNG", dpi=300)

    def _estimate_CPT(self):
        """
        Estimates the conditional probability tables associated with each node in the
        Bayesian Network.
        """

        self.bn_model = BayesianModel(list(self.structure_model.edges))
        self.cpt_model = BayesianEstimator(self.bn_model, self.filtered_df)

        for node in list(self.bn_model.nodes):
            self.bn_model.add_cpds(self.cpt_model.estimate_cpd(node))

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

        # Estimate CPTs
        self._estimate_CPT()

        if self.verbose:
            print(f"Calculating influence of all nodes on target node")

        if not self.bn_model.check_model():
            print(
                """
                There is a problem with your network structure. You have disconnected nodes
                or separated sub-networks. Please examine your network plot and re-learn your
                network structure with tweaked settings.
                """
            )
            return None

        if self.target_variable not in self.bn_model.nodes:
            print(
                """
                Your target variable has no parent nodes! Can't perform inference! Please examine
                your network plot and re-learn your network structure with tweaked settings.
                """
            )
            return None

        # Prep for belief propagation
        belief_propagation = BeliefPropagation(self.bn_model)
        belief_propagation.calibrate()

        # Iterate over all intervention nodes and values, calculating odds ratios w.r.t target variable
        overall_dict = {}

        variables_to_test = list(
            set(list(self.bn_model.nodes)) - set(list(self.target_variable))
        )

        for node in variables_to_test:
            results = []
            for value in self.filtered_df[node].unique():
                prob = belief_propagation.query(
                    variables=[self.target_variable],
                    evidence={node: value},
                    show_progress=False,
                ).values
                results.append([node, value, prob[0], prob[1]])

            results_df = pd.DataFrame(
                results, columns=["node", "value", "probability_0", "probability_1"]
            )
            results_df["odds_1"] = (
                results_df["probability_1"] / results_df["probability_0"]
            )
            results_df = results_df.sort_values(
                "value", ascending=True, inplace=False
            ).reset_index(drop=True)

            overall_dict[node] = results_df

        final_df_list = []

        for node, temp_df in overall_dict.items():
            first_value = temp_df["odds_1"].iloc[0]
            temp_df["odds_ratio"] = (temp_df["odds_1"] / first_value).round(3)
            final_df_list.append(temp_df)

        final_df = pd.concat(final_df_list)[["node", "value", "odds_ratio"]]
        self.odds_ratios = final_df

        if self.verbose:
            print(f"Saving forest plot to the following PNG file: {file_path}")

        # Clean up the dataframe of odds ratios so plot can have nice labels
        final_df2 = (
            pd.concat(
                [
                    final_df,
                    final_df.groupby("node")["value"]
                    .apply(lambda x: x.shift(-1).iloc[-1])
                    .reset_index(),
                ]
            )
            .sort_values(by=["node", "value"], ascending=True)
            .reset_index(drop=True)
        )
        final_df2["node"][final_df2["value"].isnull()] = np.nan
        final_df2["value"] = final_df2["value"].astype("Int32").astype(str)
        final_df2["value"].replace({np.nan: ""}, inplace=True)
        final_df3 = final_df2.reset_index(drop=True).reset_index()
        final_df3.rename(columns={"index": "vertical_index"}, inplace=True)
        final_df3["y_label"] = final_df3["node"] + " = " + final_df3["value"]
        final_df3["y_label"][final_df3["odds_ratio"] == 1.0] = (
            final_df3["y_label"] + " (ref)"
        )
        final_df3["y_label"].fillna("", inplace=True)

        # Produce large plot
        plt.clf()
        plt.title("Strength of Associations Between Interventions and Target Variable")
        plt.scatter(
            x=final_df3["odds_ratio"],
            y=final_df3["vertical_index"],
            s=70,
            color="b",
            alpha=0.5,
        )
        plt.xlabel("Odds Ratio")
        plt.axvline(x=1.0, color="red", linewidth="1.5", linestyle="--")
        plt.yticks(final_df3["vertical_index"], final_df3["y_label"])

        for _, row in final_df3.iterrows():
            if not np.isnan(row["odds_ratio"]):
                plt.plot(
                    [0, row["odds_ratio"]],
                    [row["vertical_index"], row["vertical_index"]],
                    color="black",
                    linewidth="0.4",
                )

        plt.xlim([0, final_df3["odds_ratio"].max() + 1])

        figure = plt.gcf()
        figure.set_size_inches(12, 7)

        plt.savefig(expanduser(file_path), bbox_inches="tight", format="PNG", dpi=300)
