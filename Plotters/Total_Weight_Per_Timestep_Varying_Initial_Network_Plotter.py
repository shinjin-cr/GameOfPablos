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
file = open('Total_Weight_Per_Timestep_Varying_Initial_Network.pkl', 'rb')
data = pickle.load(file)
font = {'size'   : 16}
shapes = ['s', 'o', 'd', '^', '<'] + ['o']*100
colors = ['r', 'y', 'g', 'p', 'b']

matplotlib.rc('font', **font)
plt.xlabel(r'Time Step ($t$)')
plt.ylabel('Total Weight')

for seed in sorted(data.keys()):
    (x, y, z, m, b) = data[seed]#(timestep, total_weight as a function of timestep, linear approximation, slope of linear approximation, y-intercept of linear approx)
    shape = shapes.pop(0)
    traverse_x = enumerate(x)
    xx = [timestep for (index, timestep) in traverse_x if index %20 == 0 ]
    traverse_y = enumerate(y)
    yy = [timestep for (index, timestep) in traverse_y if index %20 == 0 ]
    plt.plot(xx, yy, shape, label= r'$(H, D) = %s$'%str(seed))
    plt.plot(x, y)
    plt.plot(x, z, '--', label = r'$y =  %1.2f+ %1.2fx$'%(b,m))# m should always be non-negative
   
    
plt.legend(loc = 'lower right')#fontsize  = 20)
filename_for_pdf = '/'+'Total_Weight_Per_Timestep_Varying_Initial_Network.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
#######################################################################################