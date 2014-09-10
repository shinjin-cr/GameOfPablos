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

##########################################################################################
fig = plt.figure()#figsize = (15 ,10))
file = open('Degree_Distribution_with_Max_Nodes_Increased.pkl', 'rb')
data = pickle.load(file)

font = {'size'   : 16}
matplotlib.rc('font', **font)
##########################################################################################
for max_allowable_nodes in sorted(data.keys()):
    degree_dict = data[int(max_allowable_nodes)]['degree_dict']
    exper_dict  = data[int(max_allowable_nodes)]['exper_dict']
    degrees = np.array(sorted(degree_dict.keys()))
    degree_freq = np.array([float(degree_dict[d])/(exper_dict[d]*max_allowable_nodes) for d in degrees])

    #PLOT###################################################################################   
    plt.xlabel(r'Degree')#, fontsize = 20)
    plt.ylabel(r'Degree Density (Log Scale)')#, fontsize = 20)
    plt.title(r'Degree Distribution')#, fontsize = 20)
    plt.plot((degrees), np.log(degree_freq), 'o')
    plt.plot((degrees), np.log(degree_freq), '--', label = '$N = %s$'%str(int(max_allowable_nodes)))

degrees1 = degrees[1:-2]
log_degrees = np.log(degree_freq[1:-2])
m, b = np.polyfit((degrees1), log_degrees, 1)
lin_approx = m*degrees[1:-1] + b
plt.plot(degrees[1:-1], lin_approx, '--',label = r'$y = %1.1f x + %1.1f$'%(m, b))

plt.legend()
filename_for_pdf = '/'+'Degree_Distribution_with_Max_Nodes_Increased.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
#######################################################################################