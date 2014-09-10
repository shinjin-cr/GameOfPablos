
import numpy as np
#import matplotlib.pyplot as plt
import os
from random import choice 
from numpy.random import binomial
#from sympy import Eq, Symbol, solve, nsolve
import datetime
import random
import scipy.stats as ss
import networkx as nx
from PTree import *
from GrowthExperiments import *
from numpy.random import poisson as pois
import shutil
import pickle
from scipy.stats import gamma
import multiprocessing as mp
import time


""" 
The clusters as school are dual quad-cores--that means 8 jobs can be run in parallel!
"""

#######################################################################################################################################################################
arrival_distribution             = constant_arrival    # constant_arrival 
clustering                       = True
preference                       = no_more_leaves      #1/(distance_to_closest_leaf + 1)

processes = []

#######################################################################################################################################################################
#######################################################################################################################################################################
#DEGREE AND HEIGHT###################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

###
#1#
###

#MAX NODES#############################################################################################################################################################
max_allowable_nodes_interval     = [2500, 5000, 7500]
number_of_experiments            = 100
arrival_parameters               = (3,)                # constant_arrval = 3
seed                             = (4, 2)

kwargs = dictionary_utility(max_allowable_nodes_interval = max_allowable_nodes_interval, \
                            number_of_experiments = number_of_experiments, \
                            arrival_distribution = arrival_distribution, \
                            arrival_parameters = arrival_parameters, \
                            preference= preference, \
                            seed = seed, \
                            )
processes.append(mp.Process(target = height_v_mean_degree_varying_maximum_number_of_nodes_experiment, kwargs = kwargs))
#######################################################################################################################################################################

###
#2#
###


#ARRIVAL RATE###########################################################################################################################################################
max_allowable_nodes              = 5000
number_of_experiments            = 100
seed                             = (1, 2)
arrival_parameters_interval      = [1, 5, 25]

kwargs = dictionary_utility(arrival_parameters_interval  = arrival_parameters_interval, \
                            number_of_experiments = number_of_experiments, \
                            arrival_distribution  = arrival_distribution, \
                            seed = seed, \
                            preference = preference, \
                            max_allowable_nodes = max_allowable_nodes \
                            )
processes.append(mp.Process(target = height_v_mean_degree_varying_arrival_rate_experiment, kwargs = kwargs))
#######################################################################################################################################################################

###
#3#
###

#SEEDS####################################################################################################################################################################
seed_list                        = [(k, 2) for k in [1, 3, 9]] + [(4, k) for k in [1, 2, 4]] # (height, degree)
arrival_parameters               = (3,)                         # constant_arrval = 3
number_of_experiments            = 100
max_allowable_nodes              = 5000

kwargs = dictionary_utility(arrival_distribution = arrival_distribution, \
                            preference = preference, \
                            max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            arrival_parameters = arrival_parameters, \
                            seed_list = seed_list\
                            )
processes.append(mp.Process(target = height_v_mean_degree_varying_initial_network_experiment, kwargs = kwargs))
#######################################################################################################################################################################





#######################################################################################################################################################################
#######################################################################################################################################################################
#LEAF DENSITY##########################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

###
#4#
###

#MAX NUM OF NODES######################################################################################################################################################
max_allowable_nodes_interval     = [2500, 5000, 7500]
number_of_experiments            = 100
arrival_parameters               = (3,)
seed                             = (3, 3)
kwargs = dictionary_utility(arrival_parameters              = arrival_parameters, \
                            number_of_experiments           = number_of_experiments, \
                            arrival_distribution            = arrival_distribution, \
                            seed                            = seed, \
                            preference                      = preference, \
                            max_allowable_nodes_interval    = max_allowable_nodes_interval \
                            )

processes.append(mp.Process(target = leaf_density_and_node_density_varying_maximum_number_of_nodes_experiment, kwargs = kwargs))
#######################################################################################################################################################################

#######################################################################################################################################################################
#######################################################################################################################################################################
#WEIGHT WATCHER########################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

###
#5#
###

#ARRIVAL RATE###########################################################################################################################################################
max_allowable_nodes              = 5000
number_of_experiments            = 100
seed                             = (1, 2)
arrival_parameters_interval      = [1, 10, 50, 100]

kwargs = dictionary_utility(arrival_parameters_interval  = arrival_parameters_interval, \
                            number_of_experiments = number_of_experiments, \
                            arrival_distribution  = arrival_distribution, \
                            seed = seed, \
                            preference = preference, \
                            max_allowable_nodes = max_allowable_nodes \
                            )
processes.append(mp.Process(target = total_weight_per_timestep_varying_arrival_rate_experiment, kwargs = kwargs))
##########################################################################################################################################################################

###
#6#
###

#SEEDS####################################################################################################################################################################
seed_list                        = [(k, 2) for k in [1, 3, 9]] + [(4, k) for k in [1, 2, 4]] # (height, degree)
arrival_parameters               = (3,)                         # constant_arrval = 3
number_of_experiments            = 100
max_allowable_nodes              = 5000

kwargs = dictionary_utility(arrival_distribution = arrival_distribution, \
                            preference = preference, \
                            max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            arrival_parameters = arrival_parameters, \
                            seed_list = seed_list\
                            )
processes.append(mp.Process(target = total_weight_per_timestep_varying_initial_network_experiment, kwargs = kwargs))
#######################################################################################################################################################################

#######################################################################################################################################################################
#######################################################################################################################################################################
#MAX LEAF DISTANCE#####################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

###
#7#
###

#EXPERIMENT##########################################################################################################################################################
max_allowable_nodes            = 100000
seed                           = (1,2)
arrival_parameters             = (3,) 
number_of_experiments          = 4

kwargs = dictionary_utility(arrival_parameters  = arrival_parameters, \
                            arrival_distribution  = arrival_distribution, \
                            seed = seed, \
                            preference = preference, \
                            number_of_experiments = number_of_experiments,\
                            max_allowable_nodes = max_allowable_nodes \
                            ) 
processes.append(mp.Process(target = max_leaf_distance_experiment, kwargs = kwargs))
#######################################################################################

###
#8#
###

#ARRIVAL RATE##########################################################################################################################################################
max_allowable_nodes            = 100000
seed                           = (1,2)
arrival_parameters_interval    = [1, 10, 100] 

kwargs = dictionary_utility(arrival_parameters_interval  = arrival_parameters_interval, \
                            arrival_distribution  = arrival_distribution, \
                            seed = seed, \
                            preference = preference, \
                            max_allowable_nodes = max_allowable_nodes \
                            ) 
processes.append(mp.Process(target = max_leaf_distance_varying_arrival_rate_experiment, kwargs = kwargs))
#######################################################################################


for process in processes:
    process.start()
    time.sleep(1)

for process in processes:
    process.join()
    
