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
fig = plt.figure()#figsize = (10 ,10))
file = open('Total_Weight_per_Timestep.pkl', 'rb')
data = pickle.load(file)
font = {'size'   : 16}
shapes = ['>', 's', '^', 'd', 'p'] + ['o']*100
colors = ['r', 'y', 'g', 'p', 'b']

matplotlib.rc('font', **font)

#(timestep, total_weight as a function of timestep, linear approximation, slope of linear approximation, y-intercept of linear approx)
(x, y, z, m, b) = data

plt.plot(x, y, label= 'Total Weight')
plt.plot(x, z, label = r'$y =  %1.2f+ %1.2fx$'%(b,m))# m should always be non-negative
plt.xlabel(r'Time Step ($t$)')
plt.ylabel('Total Weight')
plt.legend()
    
plt.legend()#fontsize  = 20)
filename_for_pdf = '/'+'Total_Weight_Per_Timestep.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
#######################################################################################