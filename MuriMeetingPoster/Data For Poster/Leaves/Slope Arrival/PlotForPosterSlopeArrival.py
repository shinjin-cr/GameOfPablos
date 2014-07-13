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
fig = plt.figure(figsize = (10, 10))
file = open('leaf_count.pkl', 'rb')
data = pickle.load(file)
font = {'size'   : 20}
matplotlib.rc('font', **font)

arrival_list=data['arrival_list']
slope_list = data['slope_list']  
(m, b) = data['slope_of_slopes'] 
#PLOT ARRIVAL RATE V SLOPES###############################################
fig = plt.figure()
plt.xlabel(r'Rate of New Criminal Arrival ($k$)')#, fontsize = 16)
plt.ylabel(r'Rate of Street Criminal Addition$\left(\frac{d\ell}{dt}\right)$')#, fontsize = 16)
plt.title('Rate of Street Criminal Addition')#, fontsize = 16)
m, b = np.polyfit(arrival_list, slope_list, 1)
x = np.array(arrival_list)
linear_approx = m*x + b
if b< 0:
    add_sub = '$-$'
    b = np.abs(b)
else:
    add_sub = '$+$'
string_temp = '$y = %1.2f x$'%m +'%s'%add_sub + '$%1.2f$'%b
plt.plot(x, linear_approx, '--', label = r'%s'%string_temp)
plt.plot(arrival_list, slope_list, 'o')
plt.plot(arrival_list, slope_list)
plt.legend(fontsize = 16)
filename_for_pdf = '/'+'slope_arrival_for_poster.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf")
##########################################################