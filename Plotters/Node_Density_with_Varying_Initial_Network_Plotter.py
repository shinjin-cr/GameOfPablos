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
file = open('Node_Density_Varying_Initial_Network.pkl', 'rb')
data = pickle.load(file)
font = {'size'   : 16}
shapes = ['>', 's', '^', 'd', 'p'] + ['o']*100
colors = ['r', 'y', 'g', 'p', 'b']


matplotlib.rc('font', **font)
##########################################################################################
for seed in data.keys():
    
    #OPEN PICKLE FILE#######################################################################
    ND_dict = data[seed][0]
    ND_data_list = data[seed][1]
    heights = np.array(sorted(ND_dict.keys()))
    DDen = np.array([np.mean(ND_dict[h]) for h in heights])
    std= np.array([np.std(ND_dict[h]) for h in heights])
    ########################################################################################

    #PLOT DATA##############################################################################   
    plt.xlabel(r'Distance to Kingpin')#, fontsize =16)
    plt.ylabel(r'Criminal Density ( $\rho$ )')#, fontsize = 16)
    plt.title(r'Criminal Density')#, fontsize = 16)
    fit_alpha, fit_loc, fit_beta=ss.gamma.fit(ND_data_list)
    color = colors.pop(0)
    plt.plot((heights), (DDen), shapes.pop(0), color = color , label = r'$(H,D) = %s$'%str(seed))
    plt.plot((heights), (DDen), color = color)
    
    
    #PLOT GAMMA##############################################################################   
    fit_alpha, fit_loc, fit_beta=ss.gamma.fit(ND_data_list)
    constants = (fit_alpha, fit_beta, fit_loc)
    pdf_of_gamma = gamma.pdf(heights, fit_alpha, loc = fit_loc, scale = fit_beta)
    plt.plot(heights, pdf_of_gamma, '.-.',label =r'$\sim\gamma(%2.1f, %0.2f, %2.1f)$'%constants )
    ##########################################################################################
    
plt.legend()#fontsize  = 20)
filename_for_pdf = '/'+'Node_Density_Varying_Initial_Network.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
#######################################################################################