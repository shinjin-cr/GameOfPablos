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

for file in glob.glob("*.pdf"):
    strategy = re.search(r'\d+', file).group()

   
font = {'size'   : 22}

matplotlib.rc('font', **font)

file = open('ratioOC.pkl', 'rb')
data = pickle.load(file)
probability_of_wins_list = data['probability_of_wins_list']
arrival_interval         = data['arrival_interval']
fig = plt.figure()#figsize = (8, 10))
t = plt.title(r'Strategy $S_I$')#, fontsize = 35)
t.set_y(1.02)
#plt.xlim(min_arrival_parameter - .5, (max_arrival_parameter) + .5)
plt.ylim(-.05, 1.1)
plt.ylabel(r'Probability Police Win')#, fontsize = 30)
plt.xlabel(r'New Criminal Arrival Rate ($k$)')#, fontsize = 30)
plt.plot(arrival_interval, probability_of_wins_list, 'o')
#plt.plot(arrival_interval, [.5]*len(arrival_interval), '--')
ax = plt.plot(arrival_interval, probability_of_wins_list)
filename_for_pdf = '/'+'ImprovedPlotS%d.pdf'%int(strategy)


plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )