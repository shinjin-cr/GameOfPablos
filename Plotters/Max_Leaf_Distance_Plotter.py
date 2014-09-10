import matplotlib.pyplot as plt
import os
import matplotlib
import numpy as np
import pickle

s = os.path.dirname(os.path.realpath(__file__))
os.chdir(s)



########################################################################################
font = {'size'   : 20}
matplotlib.rc('font', **font)

file = open('Max_Leaf_Distance.pkl', 'rb')
data = pickle.load(file)
y_max = max([max(xy[1])for xy in data])+1
plt.ylim(0, y_max)

#######################################################################################
lines = ['-', '--', ':', '.-.']

k= 1
for experiment in data:
    x , y = experiment
    plt.xlabel('Time Step ($t$)')
    plt.ylabel('Maximum Distance to \n'+r'Street Criminals ($D_t$)')
    plt.plot(x, y, linewidth = (k)*.75, alpha=0.5, label = r'Experiment $%3d$'%k )
    k+=1

plt.legend(loc = 'lower right')


#SAVE FILE FOR PLOTS######################################################
filename_for_pdf = '/'+'Max_Leaf_Distance.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
##########################################################################