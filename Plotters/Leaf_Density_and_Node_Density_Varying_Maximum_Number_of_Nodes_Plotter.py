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
file = open('leaf_density_and_node_density_varying_max_number_of_nodes.pkl', 'rb')
data = pickle.load(file)
font = {'size'   : 18}
matplotlib.rc('font', **font)
##########################################################################################

shapes = ['>', 's', '^', 'd', 'p'] + ['o']*100


#PLOT###################################################################################   

for max_allowable_nodes in sorted(data['leaf'].keys()):
    shape = shapes.pop(0)
    leaf_density_dict = data['leaf'][max_allowable_nodes]
    heights = np.array(sorted(leaf_density_dict.keys()))
    leaf_density = np.array([np.mean(leaf_density_dict[h]) for h in heights])
    std= np.array([np.std(leaf_density_dict[h]) for h in heights])
    plt.plot((heights), (leaf_density))  
    plt.plot((heights), (leaf_density), shape, label = r'$N = %5d$ (Leaves)'%max_allowable_nodes)
    
    ND_dict = data['node'][max_allowable_nodes][0]
    ND_data_list = data['node'][max_allowable_nodes][1]
    heights = np.array(sorted(ND_dict.keys()))
    node_density = np.array([np.mean(ND_dict[h]) for h in heights])
    std= np.array([np.std(ND_dict[h]) for h in heights])
    ########################################################################################

    #PLOT DATA##############################################################################   
    plt.xlabel(r'Distance to Kingpin')#, fontsize =16)
    plt.ylabel(r'Criminal Density ( $\rho$ )')#, fontsize = 16)
    plt.title(r'Criminal Density')#, fontsize = 16)
    #fit_alpha, fit_loc, fit_beta=ss.gamma.fit(ND_data_list)
    plt.plot((heights), (node_density), shapes.pop(0) , label = r'$N = %s$ (Nodes)'%str(max_allowable_nodes))
    plt.plot((heights), (node_density))
    
    
    #PLOT GAMMA##############################################################################   
    #fit_alpha, fit_loc, fit_beta=ss.gamma.fit(ND_data_list)
    #constants = (fit_alpha, fit_beta, fit_loc)
    #pdf_of_gamma = gamma.pdf(heights, fit_alpha, loc = fit_loc, scale = fit_beta)
    #plt.plot(heights, pdf_of_gamma, '.-.',label =r'$\sim\gamma(%2.1f, %0.2f, %2.1f)$'%constants )
    ##########################################################################################

plt.xlabel(r'Distance to Kingpin')
plt.ylabel(r'Density of Criminals without Underlings')
plt.title(r'Density of Criminals without Underlings')
plt.xlim(0 , max(heights)+1) #heights at the end will be the longest because greatest number of nodes allowed
plt.ylim(0,  max(node_density) +.1)
plt.legend()
filename_for_pdf = '/'+'leaf_density_varying_maximum_number_of_nodes.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )