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

"""
This plotter is a bit confusing due to the way the data is stored.  We reproduce the dictionary here:

Data has (keys)---> (values)
   'seed_list'              ---> arrival_list (parameters of integers that denote how many nodes added at each timestep)\n'\
   'slope_list'             ---> list of slopes indexed by arrival list corresponding to change in number of leaves each timestep\n'\
    seed (tuple of int)     ---> (x, y, m, b) (tuple) where x = timsteps, y = the number of leaves indexed by timesteps, \n'\
    (Height, Degree)              m = approximate slope of the line, b = y-intercept'
"""

########################################################################################
fig = plt.figure()
file = open('leaves_per_timestep_varying_initial_network.pkl', 'rb')
data = pickle.load(file)
font = {'size'   : 18}
matplotlib.rc('font', **font)



##########################################################################################

shapes = ['>', 's', '^', 'd', 'p'] + ['o']*100
for seed in data.keys():
    if type(seed) == tuple:# and k in [10, 50, 100]:  #see dictionary explanation above, we need to parse keys for tuples!
        x = data[seed][0]
        y = data[seed][1]
        m = data[seed][2]
        b = data[seed][3]
        y_linear_approx = m*np.array(x)+np.array(b)
        xx = [x[k] for k in range(len(x)) if k%20 == 0]
        yy = [y[k] for k in range(len(y)) if k%20 == 0]
        plt.plot(x, y)
        plt.plot(xx, yy, shapes.pop(0),label = r'$(H, D) = %s$'%str(seed))
        plt.plot(x, y_linear_approx, '--', color = 'y', label = r'$y = %1.2f x + %1.2f$'%(m, b))
        
#######################################################################################
plt.xlabel(r'Time Step ($t$)')#, fontsize = 20)
plt.ylabel(r'Number of Street Criminals ($\ell(t)$)', horizontalalignment = 'center', labelpad = 25)#, fontsize = 20)
plt.title('Street Criminal Addition')#, fontsize = 20)
plt.legend(fontsize = 18, loc = 'lower right')



#SAVE FILE FOR PLOTS######################################################
filename_for_pdf = '/'+'leaves_per_timestep_varying_initial_network.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
##########################################################################