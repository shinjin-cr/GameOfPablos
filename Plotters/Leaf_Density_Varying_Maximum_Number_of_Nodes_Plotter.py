import numpy as np
import matplotlib.pyplot as plt
import os
from random import choice 
from numpy.random import binomial
from sympy import Eq, Symbol, solve, nsolve
import datetime
import random
import scipy.stats as ss
import networkx as nx
from numpy.random import poisson as pois
import pickle
from scipy.stats import gamma
import matplotlib

s = os.path.dirname(os.path.realpath(__file__))
os.chdir(s)


########################################################################################
fig = plt.figure()
file = open('leaf_density_varying_max_number_of_nodes.pkl', 'rb')
data = pickle.load(file)
font = {'size'   : 18}
matplotlib.rc('font', **font)
##########################################################################################

shapes = ['>', 's', '^', 'd', 'p'] + ['o']*100


#PLOT###################################################################################   

for max_allowable_nodes in sorted(data.keys()):
    shape = shapes.pop(0)
    leaf_density_dict = data[max_allowable_nodes]
    heights = np.array(sorted(leaf_density_dict.keys()))
    leaf_density = np.array([np.mean(leaf_density_dict[h]) for h in heights])
    std= np.array([np.std(leaf_density_dict[h]) for h in heights])
    plt.plot((heights), (leaf_density))  
    plt.plot((heights), (leaf_density), shape, label = r'$N = %5d$'%max_allowable_nodes)

plt.xlabel(r'Distance to Kingpin')
plt.ylabel(r'Density of Criminals without Underlings')
plt.title(r'Density of Criminals without Underlings')
plt.xlim(0 , max(heights)+1) #heights at the end will be the longest because greatest number of nodes allowed
plt.ylim(0,  max(leaf_density) +.05)
plt.legend()
filename_for_pdf = '/'+'leaf_density_varying_maximum_number_of_nodes.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )