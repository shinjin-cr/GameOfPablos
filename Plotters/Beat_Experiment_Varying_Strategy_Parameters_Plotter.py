import numpy as np
import matplotlib.pyplot as plt
import os
from random import choice 
from numpy.random import binomial
from sympy import Eq, Symbol, solve, nsolve
import datetime
import random
import networkx as nx
from numpy.random import poisson as pois
import pickle
import glob
import re
import matplotlib

s = os.path.dirname(os.path.realpath(__file__))
os.chdir(s)

for file in glob.glob("*.pkl"):
    strategy = int(re.search(r'\d+', file).group())
    
strategy_to_latex_dictionary    ={0 : '$S_A(p)$', 1: '$S_I$', 3: '$S_D(q)$'}

fig = plt.figure()
font = {'size'   : 22}

shapes = ['>', 's', 'o', '^', 'd', 'p'] + ['o']*100

if strategy == 0:
    parameter_variable = 'p'
else:
    parameter_variable = 'q'
matplotlib.rc('font', **font)

file = open('Beat_Experiment_Strategy_%d_Varying_Strategy_Parameters.pkl'%strategy, 'rb')
data = pickle.load(file)
t = plt.title(r'Strategy %s'%strategy_to_latex_dictionary[strategy])#, fontsize = 35)
t.set_y(1.02)
#plt.xlim(min_arrival_parameter - .5, (max_arrival_parameter) + .5)
plt.ylim(-.05, 1.05)
plt.ylabel(r'Probability Police Win')
plt.xlabel(r'New Criminal Arrival Rate ($k$)')#, fontsize = 30)
for parameter in sorted(data.keys()):
    probability_of_wins_list = data[parameter]['probability_of_wins_list']
    print probability_of_wins_list
    shape = shapes.pop(0)
    arrival_interval         = [k[0] for k in data[parameter]['arrival_interval']]
    if strategy == 1:
        plt.plot(arrival_interval, probability_of_wins_list, marker = shape, alpha = .5, markerfacecolor = 'y', label = r'$S_I$')
    else:
        plt.plot(arrival_interval, probability_of_wins_list, marker = shape, alpha = .5, markerfacecolor = 'y', label = '$%s = %2d$'%(parameter_variable, parameter))
    plt.plot(arrival_interval, probability_of_wins_list)

plt.legend(loc = 'upper right')  
filename_for_pdf = '/'+'Beat_Experiment_Strategy_%d_Varying_Parameters.pdf'%int(strategy)
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )