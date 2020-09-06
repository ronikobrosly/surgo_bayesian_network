
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

########################################
########################################
########################################


bn = Bayes_Net(target_variable = "B", verbose = True, random_seed = 111)
bn.read_data("~/Desktop/surgo/data_5000samples.csv")
bn.learn_structure("~/Desktop/structure.csv", algorithm = 'pc', significance_level = 0.05)
bn.plot_network("~/Desktop/structure.png")
bn.plot_causal_influence("~/Desktop/forest_plot.png")



#
# belief_propagation = BeliefPropagation(bn.bn_model)
#
# belief_propagation.calibrate()
#
# belief_propagation.query(variables=['B'], evidence={'MF': 0}).values[1]
#
#
#
# overall_dict = {}
#
# variables_to_test = list(set(list(bn.bn_model.nodes)) - set(list(bn.target_variable)))
#
# for node in variables_to_test:
#     results = []
#     for value in bn.filtered_df[node].unique():
#         prob = belief_propagation.query(
#             variables=['B'],
#             evidence={node:value},
#             show_progress=False
#         ).values
#         results.append([node, value, prob[0], prob[1]])
#
#     results_df = pd.DataFrame(results, columns = ['node', 'value', 'probability_0', 'probability_1'])
#     results_df['odds_1'] = results_df['probability_1'] / results_df['probability_0']
#     results_df = results_df.sort_values('value', ascending = True, inplace = False).reset_index(drop = True)
#
#     overall_dict[node] = results_df
#
# final_df_list = []
#
# for node, temp_df in overall_dict.items():
#     first_value = temp_df['odds_1'].iloc[0]
#     temp_df['odds_ratio'] = (temp_df['odds_1'] / first_value).round(2)
#     final_df_list.append(temp_df)
#
# final_df = pd.concat(final_df_list)[['node', 'value', 'odds_ratio']]
#
#
# final_df2 = pd.concat([final_df, final_df.groupby('node')['value'].apply(lambda x: x.shift(-1).iloc[-1]).reset_index()]).sort_values(by = ['node','value'], ascending = True).reset_index(drop=True)
# final_df2['node'][final_df2['value'].isnull()] = np.nan
# final_df2['value'] = final_df2['value'].astype('Int32').astype(str)
# final_df3['value'] = final_df3['value'].replace({np.nan:''})
#
#
# # Make axis labels
# final_df3 = final_df2.reset_index(drop = True).reset_index()
# final_df3 = final_df3.rename(columns = {'index':'vertical_index'})
# final_df3['y_label'] = final_df3['node'] + ' = ' + final_df3['value']
# final_df3['y_label'].fillna('', inplace = True)
#
#
#
#
#
#
#
#
# rcParams['figure.figsize'] = 12, 7
#
# plt.clf()
# plt.title('Strength of Associations Between Interventions and Target Variable')
# plt.scatter(x = final_df3['odds_ratio'], y = final_df3['vertical_index'], s = 70, color='b', alpha = 0.5)
# plt.xlabel('Odds Ratio')
# plt.axvline(x=1.0, color='red', linewidth = '1.5', linestyle='--')
# plt.yticks(final_df3['vertical_index'], final_df3['y_label'] )
#
# for _, row in final_df3.iterrows():
#     if not np.isnan(row['odds_ratio']):
#         plt.plot([0, row['odds_ratio']], [row['vertical_index'], row['vertical_index']], color = 'black', linewidth = '0.4')
#
# plt.xlim([0, final_df3['odds_ratio'].max() + 1])
#
#
# plt.show()
