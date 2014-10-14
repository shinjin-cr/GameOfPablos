#!/usr/bin/python
"""Generate bash script that run different experiments in the background

This script will create the PursuitExperiments bash scrip and dumps0-dumps7
with different parameters for each experiment.

After running this, start the new script with
./PursuitExperiments
to start.

Make sure you add executable rights (chmod +x) to both PursuitExperiments and PursuitExperiments.py
"""

from GrowthExperiments import *
import pickle
import sys

sys.stdout = open("PursuitExperiments", "w")
print("#!/bin/bash")

clustering                       = True

processes = []
max_allowable_nodes              = 3000
number_of_experiments            = 10000
max_number_of_allowed_rounds     = 7000

#######################################################################################################################################################################
#######################################################################################################################################################################
#VARYING STRATEGY PARAMETERS###########################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

###
#1#
###

#######################################################################################################################################################################
#S_A(p) aka Strategy 0#################################################################################################################################################
#######################################################################################################################################################################

#############
strategy                         = 0 #S_A(p)--investigate p-times and then arrest
#############


seed                             = (4, 2)
min_arrival_parameter            = 2
max_arrival_parameter            = 100
arrival_step_size                = 1
parameters                       = [1, 2, 4, 6, 8, 10, 20]

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed = seed, \
                            strategy = strategy,\
                            parameters = parameters,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
pickle.dump(kwargs, open("dumps0","w"))
print("nohup ./PursuitExperiments.py beat_experiment_varying_strategy_parameters_experiment dumps0 &")
print("sleep 1")

#######################################################################################################################################################################

###
#2#
###

#######################################################################################################################################################################
#S_D(q) aka Strategy 3#################################################################################################################################################
#######################################################################################################################################################################

############
strategy                         = 3 #S_D(q)--don't arrest until node of degree q is reached
############

seed                             = (4, 2)
min_arrival_parameter            = 2
max_arrival_parameter            = 100
arrival_step_size                = 1
parameters                       = [2, 4, 8, 16]

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed = seed, \
                            strategy = strategy,\
                            parameters = parameters,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
pickle.dump(kwargs, open("dumps1","w"))
print("nohup ./PursuitExperiments.py beat_experiment_varying_strategy_parameters_experiment dumps1 &")
print("sleep 1")

####################################################################################################################################################################### 


#######################################################################################################################################################################
#######################################################################################################################################################################
#INITIAL_NETWORK$$$$$$$$$$$$###########################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

###
#3#
###

#######################################################################################################################################################################
#S_A(p) aka Strategy 0#################################################################################################################################################
#######################################################################################################################################################################

#############
strategy                         = 0 #S_A(p)--investigate p-times and then arrest
strategy_parameter               = 2 #investigate twice and then arrest
#############

seed_list                        = [(k, 2) for k in [2, 4, 6]]   #(Height, Degree)
min_arrival_parameter            = 2
max_arrival_parameter            = 100
arrival_step_size                = 1

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed_list = seed_list, \
                            strategy = strategy,\
                            strategy_parameter = strategy_parameter,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
pickle.dump(kwargs, open("dumps2","w"))
print("nohup ./PursuitExperiments.py beat_experiment_varying_initial_network_experiment dumps2 &")
print("sleep 1")


###
#4#
###


#############
strategy                         = 0 #S_A(p)--investigate p-times and then arrest
strategy_parameter               = 2 #investigate twice and then arrest
#############

seed_list                        = [(4, k) for k in [1, 2, 4]]   #(Height, Degree)
min_arrival_parameter            = 2
max_arrival_parameter            = 100
arrival_step_size                = 1

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed_list = seed_list, \
                            strategy = strategy,\
                            strategy_parameter = strategy_parameter,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
pickle.dump(kwargs, open("dumps3","w"))
print("nohup ./PursuitExperiments.py beat_experiment_varying_initial_network_experiment dumps3 &")
print("sleep 1")

#######################################################################################################################################################################
#S_D(q) aka Strategy 3#################################################################################################################################################
#######################################################################################################################################################################

###
#5#
###

#############
strategy                         = 3 #S_A(p)--investigate p-times and then arrest
strategy_parameter               = 4 #degree q threshold
#############

seed_list                        = [(k, 2) for k in [2, 4, 6]]   #(Height, Degree)
min_arrival_parameter            = 2
max_arrival_parameter            = 100
arrival_step_size                = 1

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed_list = seed_list, \
                            strategy = strategy,\
                            strategy_parameter = strategy_parameter,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
pickle.dump(kwargs, open("dumps4","w"))
print("nohup ./PursuitExperiments.py beat_experiment_varying_initial_network_experiment dumps4 &")
print("sleep 1")

###
#6#
###


#############
strategy                         = 3 #S_D(q)--don't arrest until find node of degree q
strategy_parameter               = 4 #degree q threshold
#############

seed_list                        = [(4, k) for k in [1, 2, 4]]   #(Height, Degree)
min_arrival_parameter            = 2
max_arrival_parameter            = 100
arrival_step_size                = 1

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed_list = seed_list, \
                            strategy = strategy,\
                            strategy_parameter = strategy_parameter,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
pickle.dump(kwargs, open("dumps5","w"))
print("nohup ./PursuitExperiments.py beat_experiment_varying_initial_network_experiment dumps5 &")
print("sleep 1")

#######################################################################################################################################################################
#S_I aka Strategy 1#################################################################################################################################################
#######################################################################################################################################################################

###
#7#
###

#############
strategy                         = 1 #S_I, investigate until the kinpin is found!
strategy_parameter               = None
#############

seed_list                        = [(k, 2) for k in [2, 4, 6]]   #(Height, Degree)
min_arrival_parameter            = 2
max_arrival_parameter            = 100
arrival_step_size                = 1

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed_list = seed_list, \
                            strategy = strategy,\
                            strategy_parameter = strategy_parameter,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
pickle.dump(kwargs, open("dumps6","w"))
print("nohup ./PursuitExperiments.py beat_experiment_varying_initial_network_experiment dumps6 &")
print("sleep 1")

###
#8#
###


#############
strategy                         = 1 #S_I, investigate until the kinping is found
strategy_parameter               = None
#############

seed_list                        = [(4, k) for k in [1, 2, 4]]   #(Height, Degree)
min_arrival_parameter            = 2
max_arrival_parameter            = 100
arrival_step_size                = 1

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed_list = seed_list, \
                            strategy = strategy,\
                            strategy_parameter = strategy_parameter,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
pickle.dump(kwargs, open("dumps7","w"))
print("nohup ./PursuitExperiments.py beat_experiment_varying_initial_network_experiment dumps7 &")
print("sleep 1")

#######################################################################################################################################################################
