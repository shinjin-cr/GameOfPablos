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
fig = plt.figure(figsize = (10, 7))
file = open('leaf_count.pkl', 'rb')
data = pickle.load(file)
font = {'size'   : 20}
matplotlib.rc('font', **font)
##########################################################################################
for k in data.keys():
    if type(k) == int:
        x = data[k][0]
        y = data[k][1]
        plt.xlabel(r'Time Step ($t$)')#, fontsize = 20)
        plt.ylabel(r'Total Number of Street Criminals ($\ell(t)$)')#, fontsize = 20)
        plt.title('Street Criminal Addition')#, fontsize = 20)
        plt.plot(x, y, label = r'$k = %s$'%str(k))
plt.legend(fontsize = 20)
#######################################################################################
#SAVE FILE FOR PLOTS######################################################
filename_for_pdf = '/'+'leaves_per_timestep_poster.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
##########################################################################