import numpy as np
import os
#import matplotlib.pyplot as plt
from random import choice 
from numpy.random import binomial
#from sympy import Eq, Symbol, solve, nsolve
import datetime
import random
import networkx as nx
from PTree import *
from numpy.random import poisson as pois
import pickle

##################################################################

def k_star_experiment(**kwargs):
    """
    We organize the experiments in terms of k^* which is the quantity that relates officers
    sent out to the arrival rate.
    
        arrival_rate      = k
        officers_sent_out = n
        ------>         n = k - k^* 
    
    Note when k^* = -1, the officers will always win!
    
    The plot outputs the:
        k^star                          (x-axis)
        Probability Police Unit Wins    (y-axis)
    """
    ################################
    degree = 5
    height = 2
    number_of_leaves = 2**5
    ################################
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 500)
    number_of_experiments   = kwargs.pop('number_of_experiments', 10)
    arrival_parameters      = kwargs.setdefault('arrival_parameters', (10,) )                    # constant_arrval = 3
    seed                    = kwargs.setdefault('seed', (height, degree))                        #Height, Degree
    strategy                = kwargs.setdefault('strategy', 1)
    tree_type               = kwargs.setdefault('tree_type', PruneTree)
    
    if tree_type.__name__ == 'PruneTree':
        k_star_interval         = kwargs.pop('k_star_interval', np.arange(-1, number_of_leaves))
        k                       = number_of_leaves
    else:
        k_star_interval         = kwargs.pop('k_star_interval', np.arange(-1, arrival_parameters[0]))
        k                       = arrival_parameters[0]
    #######################################################################################
    
    #CREATING DIRECTORIES##################################################################
    """
    We create two directories within directory of the script
    Data > Degree_Distribution.  We then can save all plots, 
    data, and text to this directory.
    """
    s = os.path.dirname(os.path.realpath(__file__))
    os.chdir(s)
    dir_name = '/Data'    
    if not os.path.exists(os.getcwd()+dir_name):
        os.makedirs(os.getcwd()+dir_name)
    os.chdir(os.getcwd() + dir_name)
    date =  str(datetime.datetime.now().strftime(" %m:%d %H|%M|%S"))
    dir_name1 = '/k_star_exper' + ' (strat %s '%str(strategy) +tree_type.__name__ + ') '+ date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'The k^* experiment:\n\n'\
            +'The arrival rate (k) is fixed and officers are permitted to investigate up to k - k^*\n'\
            +'leaves, where -1 <= k^* <= k -1.\n\n'
    file.write(Info)    
    Info =  'Tree_type: %s\n'%tree_type.__name__ \
            +'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d \n'% max_allowable_nodes\
            +'Arrival parameters: %s\n'%str(arrival_parameters)\
            +'Strategy : %d\n' %strategy\
            +'Seed: (%d, %d)\n\n'%seed\
            +'Regarding kstar.pkl file:\n'\
            +'This returns a dictionary \'k_star_interval\' (key) to \'probability_of_wins_list\' (value) '
    file.write(Info)
    file.close()
    #########################################################################################
    
    number_of_wins_list        = np.zeros(len(k_star_interval))
    print 'Total iteratons: ', len(k_star_interval)
    for index, k_star in enumerate(k_star_interval):
        print '(index, k_star):', index, k_star
        kwargs['officers_sent_out'] = k - k_star  #SEE DEFINITION OF k in PARAMETERS SECTION
        for i in range(number_of_experiments):
            print 'EXP #: ', i
            print '#############'
            game_tree = PPTree(**kwargs)
            game_tree.growth_and_pursuit()
            if game_tree.PU.root_found == True:
                number_of_wins_list[index] += 1.
    probability_of_wins_list = number_of_wins_list/number_of_experiments
    
    file = open('kstar.pkl', 'wb')
    data = {'probability_of_wins_list': probability_of_wins_list,'k_star_interval': k_star_interval }
    pickle.dump(data, file)
    
    plt.title(r'$k^*$ and efficacy of strategy %d'%strategy)
    plt.xlim(min(k_star_interval)-.5, max(k_star_interval) + .5)
    plt.ylim(-.1, 1.1)
    plt.ylabel(r'Probability Police Win')
    plt.xlabel(r'$k^*$, where officers sent out $ =k -  k^*$')
    plt.plot(k_star_interval, probability_of_wins_list)
    plt.plot(k_star_interval, probability_of_wins_list, 'o')
    filename_for_pdf = '/'+'kstar%d.pdf'%strategy
    plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
    
def arrival_beat_experiment(**kwargs):
    """
    We fix the officers being sent as = 1.  Then we increase the arrival parameters.  
    
    Our plot is:
        arrival_parameter               (x-axis)
        probability Police Unit wins    (y-axis)
    """
    
    #STRATEGY KEYWORDS#####################################################################
    degree_threshold                = kwargs.setdefault('degree_threshold', None)
    strategy                        = kwargs.setdefault('strategy', 0)
    #######################################################################################
    
    #ARRIVAL KEYWORDS######################################################################
    min_arrival_parameter           = kwargs.pop('min_arrival_parameter', 3) # arrival_parameter (see below)
    max_arrival_parameter           = kwargs.pop('max_arrival_parameter', 50) # arrival_parameter (see below)
    arrival_step_size               = kwargs.pop('arrival_step_size', 1)
    #######################################################################################
    
    #OTHER ARGUMENTS####################################################################
    number_of_experiments           = kwargs.pop('number_of_experiments', 100)
    max_allowable_nodes             = kwargs.setdefault('max_allowable_nodes', 5)
    seed                            = kwargs.setdefault('seed', (5,2))  # (height, degree)
    tree_type                       = kwargs.setdefault('tree_type', PTree)
    #######################################################################################
    
    #OFFICERS SENT OUT MUST BE 1###########################################################
    officers_sent_out               = kwargs.setdefault('officers_sent_out', 1)
    #######################################################################################
    
    #OFFICERS SENT OUT MUST BE 1###########################################################
    clustering                      = kwargs.pop('clustering', True)
    #######################################################################################
    
    #CREATING ARRIVAL INTERVAL#############################################################
    #list goes from 2 to max_arrival_parameter
    k = (max_arrival_parameter-min_arrival_parameter)/arrival_step_size +1
    arrival_interval = [0]*(k)
    for j in range(k):
        arrival_interval[j] = (min_arrival_parameter+j*arrival_step_size,) 
    #######################################################################################
    
    #CREATING DIRECTORIES##################################################################
    """
    We create two directories within directory of the script
    Data > Degree_Distribution.  We then can save all plots, 
    data, and text to this directory.
    """
    s = os.path.dirname(os.path.realpath(__file__))
    os.chdir(s)
    dir_name = '/Data'    
    if not os.path.exists(os.getcwd()+dir_name):
        os.makedirs(os.getcwd()+dir_name)
    os.chdir(os.getcwd() + dir_name)
    date =  str(datetime.datetime.now().strftime(" %m:%d %H|%M|%S"))
    dir_name1 = '/Ratio_Experiment' + ' (strat %s)'%str(strategy) + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'The ratio experiment:\n\n'\
            +'The arrival rate (k) is increased while a single officers is sent out\n'
    file.write(Info)    
    Info =   'Tree Type: %s \n'%tree_type.__name__ \
            +'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d \n'% max_allowable_nodes\
            +'Max arrival parameter (goes from 2, ... , parameter): %s\n'%str(max_arrival_parameter)\
            +'Strategy : %d\n' %strategy\
            +'Seed: (%d, %d)\n\n'%seed\
            +'Regarding ratioOC.pkl file:\n'\
            +'This returns a dictionary with keys \'arrival_interval\' (key/x-axis) to \'probability_of_wins_list\' (key/y-axis), whose values are lists of such. '\
            +'Degree Threshold: %s'%str(degree_threshold)
    file.write(Info)
    file.close()
    #########################################################################################
    
    number_of_wins_list        = np.zeros(len(arrival_interval))
    
    #LOG TITLE################################################################################
    if clustering:
        logger = open('log_for_beat_s%d.txt'%strategy, 'wb')
        logger.write('<<<<Beat Log>>>>\n\n\nTotal Iterations %3d\n'%len(arrival_interval))
        logger.close()
    #########################################################################################
    
    for index, arrival_rate in enumerate(arrival_interval):
        kwargs['arrival_parameters'] = arrival_rate
        if clustering:
            logger = open('log_for_beat_s%d.txt'%strategy, 'ab')
            logger.write('\n\n(index, arrival rate): %3d%3d\n'%(index, arrival_rate[0]))
            logger.write('######################\n\n')
            logger.close()
        else:
            print '(index, arrival_rate):', index, arrival_rate[0]
            print arrival_rate
        for i in range(number_of_experiments):
            if clustering:
                logger = open('log_for_beat_s%d.txt'%strategy, 'ab')
                logger.write('EXP # %5d\n'%i)
                logger.close()
            else:
                print 'EXP #: ', i
                print '#############'
            game_tree = PPTree(**kwargs)
            game_tree.growth_and_pursuit()
            if game_tree.PU.root_found == True:
                number_of_wins_list[index] += 1.
    probability_of_wins_list = number_of_wins_list/number_of_experiments
    file = open('ratioOC%d.pkl'%int(strategy), 'wb')
    data = {'probability_of_wins_list': probability_of_wins_list,'arrival_interval': arrival_interval }
    pickle.dump(data, file)
    file.close()
    
     

def the_beat_plot(**kwargs):
    """
    We fix the officers being sent as = 1.  Then we increase the arrival parameters.  
    
    Our plot is:
        arrival_parameter               (x-axis)
        probability Police Unit wins    (y-axis)
    """
    
    #STRATEGY KEYWORDS#####################################################################
    degree_threshold                = kwargs.setdefault('degree_threshold', None)
    strategy                        = kwargs.setdefault('strategy', 0)
    #######################################################################################
    
    #ARRIVAL KEYWORDS######################################################################
    min_arrival_parameter           = kwargs.pop('min_arrival_parameter', 1) # arrival_parameter (see below)
    max_arrival_parameter           = kwargs.pop('max_arrival_parameter', 25) # arrival_parameter (see below)
    arrival_step_size               = kwargs.pop('arrival_step_size', 1)
    #######################################################################################
    
    #OTHER ARGUMENTS####################################################################
    number_of_experiments           = kwargs.pop('number_of_experiments', 1000)
    max_allowable_nodes             = kwargs.setdefault('max_allowable_nodes', 5000)
    seed                            = kwargs.setdefault('seed', (5,2))  # (height, degree)
    tree_type                       = kwargs.setdefault('tree_type', PTree)
    #######################################################################################
    
    #OFFICERS SENT OUT MUST BE 1###########################################################
    officers_sent_out               = kwargs.setdefault('officers_sent_out', 1)
    #######################################################################################
    
    #CREATING ARRIVAL INTERVAL#############################################################
    #list goes from 2 to max_arrival_parameter
    k = (max_arrival_parameter-min_arrival_parameter)/arrival_step_size +1
    arrival_interval = [0]*(k)
    for j in range(k):
        arrival_interval[j] = (min_arrival_parameter+j*arrival_step_size,) 
    #######################################################################################
    
    
    file = open('ratioOC.pkl', 'rb')
    data = pickle.load(file)
    probability_of_wins_list = data['probability_of_wins_list']
    arrival_interval         = data['arrival_interval']
    
    fig = plt.figure()
    plt.title(r'Efficacy of Strategy %d'%strategy)
    plt.xlim(min_arrival_parameter - .5, (max_arrival_parameter) + .5)
    plt.ylim(-.05, 1.1)
    plt.ylabel(r'Probability Police Win')
    plt.xlabel(r'Arrival Rate (k)')
    plt.plot(arrival_interval, probability_of_wins_list, 'o')
    plt.plot(arrival_interval, probability_of_wins_list)
    filename_for_pdf = '/'+'ratioOC Str%d.pdf'%strategy
    plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )


def arrival_beat_experiment_with_varying_parameters(**kwargs):
    """
    We fix the officers being sent as = 1.  Then we increase the arrival parameters.  
    
    Our plot is:
        arrival_parameter               (x-axis)
        probability Police Unit wins    (y-axis)
    """
    
    #STRATEGY KEYWORDS#####################################################################
    strategy                        = kwargs.setdefault('strategy', 3)
    #######################################################################################
    
    #ARRIVAL KEYWORDS######################################################################
    min_arrival_parameter           = kwargs.pop('min_arrival_parameter', 10) # arrival_parameter (see below)
    max_arrival_parameter           = kwargs.pop('max_arrival_parameter', 30) # arrival_parameter (see below)
    arrival_step_size               = kwargs.pop('arrival_step_size', 1)
    #######################################################################################
    
    #OTHER ARGUMENTS####################################################################
    number_of_experiments           = kwargs.pop('number_of_experiments', 100)
    max_allowable_nodes             = kwargs.setdefault('max_allowable_nodes', 5000)
    seed                            = kwargs.setdefault('seed', (3,2))  # (height, degree)
    tree_type                       = kwargs.setdefault('tree_type', PTree)
    #######################################################################################
    
    #OFFICERS SENT OUT MUST BE 1###########################################################
    officers_sent_out               = kwargs.setdefault('officers_sent_out', 1)
    #######################################################################################
    
    #OFFICERS SENT OUT MUST BE 1###########################################################
    clustering                      = kwargs.pop('clustering', True)
    #######################################################################################
    
    #PARAMETERS###########################################################
    if strategy == 0:
        parameters = kwargs.pop('investigation_interval', range(5, 1, -1))
    elif strategy == 3:
        parameters = kwargs.pop('degree_threshold_interval', range(6, 2, -1))
    else:
        parameters = [1]
    #######################################################################################
    
    #CREATING ARRIVAL INTERVAL#############################################################
    #list goes from 2 to max_arrival_parameter
    k = (max_arrival_parameter-min_arrival_parameter)/arrival_step_size +1
    arrival_interval = [0]*(k)
    for j in range(k):
        arrival_interval[j] = (min_arrival_parameter+j*arrival_step_size,) 
    #######################################################################################
    
    #CREATING DIRECTORIES##################################################################
    """
    We create two directories within directory of the script
    Data > Degree_Distribution.  We then can save all plots, 
    data, and text to this directory.
    """
    s = os.path.dirname(os.path.realpath(__file__))
    os.chdir(s)
    dir_name = '/Data'    
    if not os.path.exists(os.getcwd()+dir_name):
        os.makedirs(os.getcwd()+dir_name)
    os.chdir(os.getcwd() + dir_name)
    date =  str(datetime.datetime.now().strftime(" %m:%d %H|%M|%S"))
    dir_name1 = '/Ratio_Experiment' + ' (strat %s)'%str(strategy) + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'The ratio experiment:\n\n'\
            +'The arrival rate (k) is increased while a single officers is sent out\n'
    file.write(Info)    
    Info =   'Tree Type: %s \n'%tree_type.__name__ \
            +'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d \n'% max_allowable_nodes\
            +'Max arrival parameter (goes from 2, ... , parameter): %s\n'%str(max_arrival_parameter)\
            +'Strategy : %d\n' %strategy\
            +'Seed: (%d, %d)\n\n'%seed\
            +'Regarding ratioOC.pkl file:\n'\
            +'This returns a dictionary with keys \'arrival_interval\' (key/x-axis) to \'probability_of_wins_list\' (key/y-axis), whose values are lists of such. '\
            +'Parmaters: %2d to %2d'%(min(parameters), max(parameters))
    file.write(Info)
    file.close()
    #########################################################################################
    
    data = {}
    
    #LOG TITLE################################################################################
    if clustering:
        logger = open('log_for_beat_s%d.txt'%strategy, 'wb')
        logger.write('<<<<Beat Log>>>>\n\n\nTotal Iterations %3d\n'%len(arrival_interval))
        logger.close()
    #########################################################################################
    
    
    for parameter in parameters:
        number_of_wins_list        = np.zeros(len(arrival_interval))
        if strategy == 0:
            kwargs['degree_threshold'] = parameter
        elif strategy == 3:
            kwargs['number_of_investigations_for_strategy_SI'] = parameter
        else:
            pass
        for index, arrival_rate in enumerate(arrival_interval):
            kwargs['arrival_parameters'] = arrival_rate
            if clustering:
                logger = open('log_for_beat_s%d.txt'%strategy, 'ab')
                logger.write('\n\n(index, arrival rate): %3d%3d\n'%(index, arrival_rate[0]))
                logger.write('######################\n\n')
                logger.close()
            else:
                print '(index, arrival_rate):', index, arrival_rate[0]
                print arrival_rate
            for i in range(number_of_experiments):
                if clustering:
                    logger = open('log_for_beat_s%d.txt'%strategy, 'ab')
                    logger.write('EXP # %5d\n'%i)
                    logger.close()
                else:
                    print 'EXP #: ', i
                    print '#############'
                game_tree = PPTree(**kwargs)
                game_tree.growth_and_pursuit()
                if game_tree.PU.root_found == True:
                    number_of_wins_list[index] += 1.
            probability_of_wins_list = number_of_wins_list/number_of_experiments
            data[parameter] = {'probability_of_wins_list': probability_of_wins_list,'arrival_interval': arrival_interval }
            file = open('ChangingParametersOfficerOC%d.pkl'%int(strategy), 'wb')
            pickle.dump(data, file)
            file.close()
    
if __name__=='__main__':
    arrival_beat_experiment_with_varying_parameters()
        
    #ratio_of_officers_to_new_criminals_experiment()
        