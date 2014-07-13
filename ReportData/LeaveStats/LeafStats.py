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
fig = plt.figure( figsize = (16, 6))
file = open('leaf_count.pkl', 'rb')
data = pickle.load(file)
font = {'size'   : 20}
matplotlib.rc('font', **font)


left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.8      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.23   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                wspace=wspace, hspace=wspace)

##########################################################################################
plt.subplot(1, 2, 1)
for k in data.keys():
    if type(k) == int and k in [10, 50, 100]:
        x = data[k][0]
        y = data[k][1]
        plt.plot(x, y, label = r'$k = %s$'%str(k))
#######################################################################################
plt.xlabel(r'Time Step ($t$)')#, fontsize = 20)
plt.ylabel('Number of\n ' +r' Street Criminals ($\ell(t)$)', horizontalalignment = 'center', labelpad = 25)#, fontsize = 20)
plt.title('Street Criminal \n Addition')#, fontsize = 20)
plt.legend(fontsize = 18)


arrival_list=data['arrival_list']
slope_list = data['slope_list']  
(m, b) = data['slope_of_slopes'] 
#PLOT ARRIVAL RATE V SLOPES###############################################
plt.subplot(1, 2, 2)
plt.xlabel(r'New Criminal Arrival Rate ($k$)')#, fontsize = 16)
plt.ylabel('Rate of Street \n'+r'Criminal Addition $\left(\frac{d\ell}{dt}\right)$', horizontalalignment = 'center', labelpad = 27)#, fontsize = 16)
plt.title('Rate of Street \n Criminal Addition')#, fontsize = 16)
m, b = np.polyfit(arrival_list, slope_list, 1)
x = np.array(arrival_list)
plt.xlim( min(x) -2,max(x)+1 )
linear_approx = m*x + b
if b< 0:
    add_sub = '$-$'
    b = np.abs(b)
else:
    add_sub = '$+$'
string_temp = r'$\frac{d \ell}{d t} = %1.2f k$'%m +'%s'%add_sub + '$%1.2f$'%b
plt.plot(x, linear_approx, '--', label = r'%s'%string_temp)
plt.plot(arrival_list, slope_list, 'o')
plt.plot(arrival_list, slope_list)
plt.legend(loc = 'lower right', fontsize = 18)
#fig.set_tight_layout(True)
#SAVE FILE FOR PLOTS######################################################
filename_for_pdf = '/'+'leafstats.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
##########################################################################