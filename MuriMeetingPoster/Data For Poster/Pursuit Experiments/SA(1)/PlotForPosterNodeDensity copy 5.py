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

########################################################################################
fig = plt.figure()#figsize = (15 ,10))
file = open('ND.pkl', 'rb')
data = pickle.load(file)
font = {'size'   : 18}

matplotlib.rc('font', **font)
##########################################################################################
for max_allowable_nodes_pair in data.keys():
    max_allowable_nodes_pair = (max_allowable_nodes_pair[0], int(max_allowable_nodes_pair[1]))
    #OPEN PICKLE FILE#######################################################################
    ND_dict = data[max_allowable_nodes_pair][0]
    ND_data_list = data[max_allowable_nodes_pair][1]
    heights = np.array(sorted(ND_dict.keys()))
    DDen = np.array([np.mean(ND_dict[h]) for h in heights])
    std= np.array([np.std(ND_dict[h]) for h in heights])
    ########################################################################################

    #PLOT DATA##############################################################################   
    plt.xlabel(r'Distance to Root', fontsize =20)
    plt.ylabel(r'Node Density', fontsize = 20)
    plt.title(r'Node Density', fontsize = 20)
    fit_alpha, fit_loc, fit_beta=ss.gamma.fit(ND_data_list)
    plt.plot((heights), (DDen), 'o')
    plt.errorbar((heights), (DDen), label = r'$N = %s$'%str(max_allowable_nodes_pair[1]))#yerr= std
    
    #PLOT GAMMA##############################################################################   
    fit_alpha, fit_loc, fit_beta=ss.gamma.fit(ND_data_list)
    constants = (fit_alpha, fit_beta, fit_loc)
    pdf_of_gamma = gamma.pdf(heights, fit_alpha, loc = fit_loc, scale = fit_beta)
    plt.plot(heights, pdf_of_gamma, '.-.',label =r'$\sim\gamma(%2.1f, %0.2f, %2.1f)$'%constants )
    ##########################################################################################
    
plt.legend(fontsize  = 20)
filename_for_pdf = '/'+'NodeDensity_Poster.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
#######################################################################################