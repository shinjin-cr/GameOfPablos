import numpy as np
import matplotlib.pyplot as plt
import os
from random import choice 
from numpy.random import binomial
#from sympy import Eq, Symbol, solve, nsolve
import datetime
import random
import scipy.stats as ss
import networkx as nx
from numpy.random import poisson as pois
import shutil
import pickle
from scipy.stats import gamma




#CREATING DIRECTORIES##################################################################
"""
We create two directories within directory of the script
Data > Degree_Distribution.  We then can save all plots, 
data, and text to this directory.
"""
s = os.path.dirname(os.path.realpath(__file__))
os.chdir(s)
#######################################################################################


#OPEN PICKLE FILE######################################################################
file = open('Height_V_Mean_Degree_Varying_Initial_Network.pkl', 'rb')
data = pickle.load(file)
file.close()

fig = plt.figure()#figsize = (15 ,10))

shapes = ['>', 's','o', 'd', 'p'] + ['o']*100

plt.xlabel(r'Distance to Kingpin')
plt.ylabel(r'Degree (Average)')
plt.title(r'Mean Degree depending on Distance to Kingpin')


max_height = max([max(data[seed].keys()) for seed in data.keys()])
max_degree = max([max([max(degrees) for degrees in data[seed].values()]) for seed in data.keys()])
plt.xlim(0 - .1 , max_height +1)
plt.ylim(0 -.1,  max_degree +.1)

for seed in sorted(data.keys()):
    shape = shapes.pop(0)
    height_degree_dict = data[seed]
    heights = np.array(sorted(height_degree_dict.keys()))
    average_degree = np.array([np.mean(height_degree_dict[h]) for h in heights])
    std= np.array([np.std(height_degree_dict[h]) for h in heights])
    #######################################################################################

    #PLOT###################################################################################   
    plt.errorbar((heights), (average_degree))#,  yerr= std)
    plt.plot((heights), (average_degree), shape, alpha = .5, label = r'$(H, D) = (%2d, %2d)$'%seed)
    #######################################################################################
plt.legend()
filename_for_pdf = '/'+'Height_V_Mean_Degree_Varying_Initial_Network.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
