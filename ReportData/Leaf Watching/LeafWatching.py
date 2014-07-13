import matplotlib.pyplot as plt
import os
import matplotlib
import numpy as np
import pickle

s = os.path.dirname(os.path.realpath(__file__))
os.chdir(s)

file = open('leaf_watcher_arrival_increasing.pkl', 'rb')
data_dict = pickle.load(file)


########################################################################################
fig = plt.figure( figsize = (16, 6))
font = {'size'   : 20}
matplotlib.rc('font', **font)


left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.8      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.23   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                wspace=wspace, hspace=wspace)

##########################################################################################
plt.subplot(1, 2, 2)

max_y = max([max(data_dict[k][1]) for k in data_dict.keys()])
plt.ylim(0, max_y+1)
plt.xlim(0, 500)

for k in [1, 30, 90]:#sorted(data_dict.keys()):
    x , y = np.array(data_dict[k][0]), np.array(data_dict[k][1])
    print 'x =' ,x
    print 'y =', y
    plt.xlabel('Time Step ($t$)')
    #plt.ylabel(r'Maximum Distance to Street Criminals ($D_t$)')
    plt.plot(x, y,  label = r'$k =%s$ '%str(k))
plt.legend(loc = 'lower right')
#######################################################################################

plt.subplot(1, 2, 1)

file = open('leaf_watcher.pkl', 'rb')
data = pickle.load(file)
x , y = data[0], data[1]
plt.ylim(0, max(y)+.5)
plt.xlabel('Time Step ($t$)')
plt.ylabel('Maximum Distance to \n'+r'Street Criminals ($D_t$)')
plt.plot(x, y, label = r'$k = 1$')
plt.legend(loc = 'lower right')


#SAVE FILE FOR PLOTS######################################################
filename_for_pdf = '/'+'LeafWatcher.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
##########################################################################



