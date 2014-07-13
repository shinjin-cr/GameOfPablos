import numpy as np
#import matplotlib.pyplot as plt
import os
from random import choice 
from numpy.random import binomial
#from sympy import Eq, Symbol, solve, nsolve
import datetime
import random
import scipy.stats as ss
import networkx as nx
from PTree import *
from numpy.random import poisson as pois
import pickle
from scipy.stats import gamma



#DEGREE DISTRIBUTION#####################################################################################################################################################
def degree_distribution_increasing_max_nodes_experiment(**kwargs):
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes_interval     = kwargs.pop('max_allowable_nodes', np.linspace(2000, 6000, 3))
    number_of_experiments            = kwargs.pop('number_of_experiments', 100)
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters               = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference                       = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    clustering                       = kwargs.pop('clustering', True)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/DegDist_MaxNodes' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Degree Distribution as Effected By Maximum Allowable Nodes Increases:\n\n'
    file.write(Info)    
    Info = 'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes Interval: %4d, %4d\n'%(max_allowable_nodes_interval[0], max_allowable_nodes_interval[-1])\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameters: %s\n\n'%str(arrival_parameters)\
            +'Regarding Degree_Distribution.pkl file:\n'\
            +'The key \'degree_dict\' is the dictionary with key: degree and value: number of occurances \n' \
            +'in all experiments! \n'\
            +'The key \'exper_dict\' is the dictionary for the number of experiments each degree showed up\n'\
            +'The degree frequency is obtained by: [float(degree_dict[d])/exper_dict[d] for d in degrees]'
    file.write(Info)
    file.close()
    ##########################################################################################
    
    
    
    #LOGGING###################################################################################
    if clustering:
        logger = open('Log_for_DegDist_MaxNodes.txt', 'wb')
        logger.write('Degree Distribution as Max Nodes Increases' + date + '\n\n')
        logger.close()
    else:
        print 'Height V Degree Experiment'
        print '##########################'
    ############################################################################################
    
    #DATA ARRAY#################################################################################
    data = {}
    ############################################################################################
    
    #LOOP FOR PEAKS##############################################################################
    for max_allowable_nodes in max_allowable_nodes_interval:
        kwargs['max_allowable_nodes'] = int(max_allowable_nodes)
        #THE EXPERIMENT##########################################################################
        
        #LOGGING#################################################################################
        if clustering:
            logger = open('Log_for_DegDist_MaxNodes.txt', 'a')
            log_update = 'Max Nodes:%5d \n \n'%int(max_allowable_nodes) + '######################\n'
            logger.write(log_update)
            logger.close()
        else:
            print '##############################'
            print 'Max Nodes:', int(max_allowable_nodes)
        ##########################################################################################
        
        #DATA FOR EXPERIMENT######################################################################
        degree_dict = {}        # key: degree, value: frequency
        exper_dict = {}         # key: degree, value: number of experiments for degree
        ##########################################################################################
        
        for k in range(number_of_experiments):
            
            #LOGGING###############################################################################
            if clustering:
                logger = open('Log_for_DegDist_MaxNodes.txt', 'a')
                log_update = 'EXP #: %4d'%k
                logger.write(log_update + '\n')
                logger.close()
            else:
                print log_update
            ########################################################################################
            
            #GROWTH#################################################################################
            G = PTree(**kwargs)
            G.growth()
            degrees_of_G = nx.degree(G)   #key: node, value: degree
            exper_dict_temp = {}
            ########################################################################################
            
            #UPDATE DATA FOR EXPERIMENT#############################################################
            exper_dict_temp   = list(set(degrees_of_G.values()))
            
            for degree in degrees_of_G.values():
                degree_dict[degree] = 1 + degree_dict.get(degree, 0)
            for degree in degree_dict.keys():
                exper_dict[degree] = 1 + exper_dict.get(degree, 0)
            ########################################################################################
        
            #DATA STORAGE##############################################################################
            data[int(max_allowable_nodes)] = {'degree_dict': degree_dict, 'exper_dict': exper_dict}
            file = open('DegDist_MaxNodes.pkl', 'ab')
            pickle.dump(data, file)
            file.close()
            ###########################################################################################
        
    
    
    ##########################################################################################
    
def degree_distribution_increasing_max_nodes_plot(**kwargs):
    
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/DD_Peak' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    fig = plt.figure(figsize = (15 ,10))
    file = open('DegDist.pkl', 'rb')
    data = pickle.load(file)
    ##########################################################################################
    for max_allowable_nodes in max_allowable_nodes_interval:
        degree_dict = data[int(max_allowable_nodes)]['degree_dict']
        exper_dict  = data[int(max_allowable_nodes)]['exper_dict']
        degrees = np.array(sorted(degree_dict.keys()))
        degree_freq = np.array([float(degree_dict[d])/(exper_dict[d]*max_allowable_nodes) for d in degrees])
    
        #PLOT###################################################################################   
        plt.xlabel(r'Degree')
        plt.ylabel(r'Degree Density (log)')
        plt.title(r'Degree Distribution')
        #plt.xlim(min(degrees) - 1 , max(degrees)+1)
        #plt.ylim(0 -.1,  max(degree_freq) +5)
        plt.plot((degrees), np.log(degree_freq), 'o')
        plt.plot((degrees), np.log(degree_freq), label = r'$N = %s$'%str(max_allowable_nodes))
    degrees = np.delete(degrees, [0, -1, -2, -3, -4])
    log_degrees = np.delete(np.log(degree_freq), [0, -1, -2, -3, -4])
    m, b = np.polyfit((degrees), log_degrees, 1)
    lin_approx = m*degrees + b
    plt.plot(degrees, lin_approx, '--',label = r'$y = %1.1f x + %1.1f$'%(m, b))
    
    plt.legend()
    filename_for_pdf = '/'+'DegDistMaxNodes.pdf'
    plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )

def degree_distribution_increasing_arrival_rate_experiment(**kwargs):
    
    #KEYWORD ARGUMENTS####################################################################
    arrival_parameters_interval      = kwargs.pop('arrival_parameters_interval', np.concatenate(([1], np.linspace(10, 50, 5))))
    number_of_experiments            = kwargs.pop('number_of_experiments', 10)
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    preference                       = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    clustering                       = kwargs.pop('clustering', True)
    max_allowable_nodes              = kwargs.setdefault('max_allowable_nodes', 100)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/DegDist_ArrivalRate' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Degree Distribution as Effected By Arrival Rate Increase:\n\n'
    file.write(Info)    
    Info = 'Experiments:%4d \n'%number_of_experiments\
            +'Max and Min Arrival Rates: %4d, %4d\n'%(arrival_parameters_interval[0], arrival_parameters_interval[-1])\
            +'Preference: %s\n'% preference.__doc__\
            +'Max Number of Nodes: %5d\n\n'%max_allowable_nodes\
            +'Regarding DegDist_ArrivalRate.pkl file:\n'\
            +'The key \'degree_dict\' is the dictionary with key: degree and value: number of occurances \n' \
            +'in all experiments! \n'\
            +'The key \'exper_dict\' is the dictionary for the number of experiments each degree showed up\n'\
            +'The degree frequency is obtained by: [float(degree_dict[d])/exper_dict[d] for d in degrees]'
    file.write(Info)
    file.close()
    ##########################################################################################
    
    
    
    #LOGGING###################################################################################
    if clustering:
        logger = open('Log_for_DegDist_ArrivalRate.txt', 'wb')
        logger.write('Degree Distribution as Max Nodes Increases' + date + '\n\n')
        logger.write('Max and Min Arrival Rates: %3d, %3d\n'%(arrival_parameters_interval[0], arrival_parameters_interval[-1]))
        logger.write('# of iterations: %3d\n\n'%len(arrival_parameters_interval))
        logger.close()
    else:
        print 'Height V Degree Experiment'
        print '##########################'
    ############################################################################################
    
    #DATA ARRAY#################################################################################
    data = {}
    ############################################################################################
    
    #LOOP FOR PEAKS##############################################################################
    for arrival_parameter in arrival_parameters_interval:
        kwargs['arrival_parameters'] = int(arrival_parameter),
        
        #THE EXPERIMENT##########################################################################
        
        #LOGGING#################################################################################
        if clustering:
            logger = open('Log_for_DegDist_ArrivalRate.txt', 'a')
            log_update = '\n \n Arrival_Rate:%4d \n \n'%int(arrival_parameter) + '######################\n'
            logger.write(log_update)
            logger.close()
        else:
            print '##############################'
            print 'Max Nodes:', int(max_allowable_nodes)
        ##########################################################################################
        
        #DATA FOR EXPERIMENT######################################################################
        degree_dict = {}        # key: degree, value: frequency
        exper_dict = {}         # key: degree, value: number of experiments for degree
        ##########################################################################################
        
        for k in range(number_of_experiments):
            
            #LOGGING###############################################################################
            if clustering:
                logger = open('Log_for_DegDist_ArrivalRate.txt', 'a')
                log_update = 'EXP #: %4d'%k
                logger.write(log_update + '\n')
                logger.close()
            else:
                print log_update
            ########################################################################################
            
            #GROWTH#################################################################################
            G = PTree(**kwargs)
            G.growth()
            degrees_of_G = nx.degree(G)   #key: node, value: degree
            exper_dict_temp = {}
            ########################################################################################
            
            #UPDATE DATA FOR EXPERIMENT#############################################################
            exper_dict_temp   = list(set(degrees_of_G.values()))
            
            for degree in degrees_of_G.values():
                degree_dict[degree] = 1 + degree_dict.get(degree, 0)
            for degree in degree_dict.keys():
                exper_dict[degree] = 1 + exper_dict.get(degree, 0)
            ########################################################################################
        
            #DATA STORAGE##############################################################################
            data[int(max_allowable_nodes)] = {'degree_dict': degree_dict, 'exper_dict': exper_dict}
            file = open('DegDist_ArrivalRate.pkl', 'ab')
            pickle.dump(data, file)
            file.close()
            ###########################################################################################
        
    
    
    ##########################################################################################
    

#######################################################################################################################################################################

#DEGREE AND HEIGHT#####################################################################################################################################################
def height_v_degree_experiment(**kwargs):
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 20)
    number_of_experiments   = kwargs.pop('number_of_experiments', 1)
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters      = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference              = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Degree_V_Height' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Degree Distribution:\n\n'
    file.write(Info)    
    Info = 'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d\n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameters: %s\n\n'%str(arrival_parameters)\
            +'Regarding HD.pkl file:\n'\
            +'This returns a dictionary distance_to_root/height (key) to a list of average degrees indexed by each experiment(values)'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #THE EXPERIMENT##########################################################################
    print 'Height V Degree Experiment'
    print '##############################'
    HD_dict = {}
    exper_dict = {}
    file = open('HD.pkl', 'wb')
    for k in range(number_of_experiments):
        print 'EXP #: ', k
        G = PTree(**kwargs)
        G.growth()
        levels = G.get_levels()
        height = len(levels)
        for level_index in range(height):
            degree_list_temp = []
            for node in levels[level_index]:
                degree_list_temp.append(float(G.out_degree(node)))
            #HD_dict.setdefault(level_index, []).extend(degree_list_temp)
            HD_dict.setdefault(level_index, []).append(np.mean(degree_list_temp))     
            data = HD_dict
    pickle.dump(data, file)
    file.close()
    ########################################################################################
    
def plot_height_v_degree(**kwargs):
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 2000)
    number_of_experiments   = kwargs.setdefault('number_of_experiments', 3)
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters      = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference              = kwargs.setdefault('preference', no_more_leaves)   # 1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Degree_V_Height' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #######################################################################################
    
    #EXPERIMENT############################################################################
    height_v_degree_experiment(**kwargs)
    #######################################################################################
    
    #OPEN PICKLE FILE######################################################################
    file = open('HD.pkl', 'rb')
    HD_dict = pickle.load(file)
    heights = np.array(sorted(HD_dict.keys()))
    average_degree = np.array([np.mean(HD_dict[h]) for h in heights])
    std= np.array([np.std(HD_dict[h]) for h in heights])
    #######################################################################################
    
    #PLOT###################################################################################   
    fig = plt.figure(figsize = (15 ,10))
    plt.xlabel(r'Distance to Root')
    plt.ylabel(r'Degree (Average)')
    plt.title(r'Degree Distribution depending on Height')
    plt.xlim(min(heights) - .1 , max(heights)+1)
    plt.ylim(0 -.1,  max(average_degree) +.1)
    plt.errorbar((heights), (average_degree),  yerr= std)
    plt.plot((heights), (average_degree), 'o')
    filename_for_pdf = '/'+'HD.pdf'
    plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
    #######################################################################################
        
#######################################################################################################################################################################

#LEAF DISTRIBUTION#####################################################################################################################################################
def leaf_distribution(**kwargs):
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 2000)
    number_of_experiments   = kwargs.pop('number_of_experiments', 100)
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters      = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference              = kwargs.setdefault('preference', height_preference)   #1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Leaf_Distribution' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Leaf Distribution By Level:\n\n'
    file.write(Info)    
    Info = 'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d\n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameters: %s\n\n'%str(arrival_parameters)\
            +'Regarding HL.pkl file:\n'\
            +'This returns a dictionary distance_to_root/height (key) to a list of the frequency of leaves indexed by experiment(values)'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #THE EXPERIMENT##########################################################################
    print 'Leaf Distribution Experiment'
    print '##############################'
    HL_dict = {}
    exper_dict = {}
    for k in range(number_of_experiments):
        print 'EXP #: ', k
        G = PTree(**kwargs)
        G.growth()
        levels = G.get_levels()
        height = len(levels)
        HL_dict_temp = {}
        for leaf in G.leaves:
            HL_dict_temp[G.node[leaf]['height']] = 1 + HL_dict_temp.get(G.node[leaf]['height'],0)
        for key in HL_dict_temp.keys():
            HL_dict.setdefault(key, []).append(HL_dict_temp[key])     
    ##########################################################################################
    
    
    #THE DUMP#################################################################################
    file = open('HL.pkl', 'wb')
    data = HL_dict
    pickle.dump(data, file)
    file.close()
    ########################################################################################
    
    #OPEN PICKLE FILE#######################################################################
    file = open('HL.pkl', 'rb')
    HL_dict = pickle.load(file)
    heights = np.array(sorted(HL_dict.keys()))
    leaf_freq = np.array([np.mean(HL_dict[h]) for h in heights])
    std= np.array([np.std(HL_dict[h]) for h in heights])
    ########################################################################################
    
    #PLOT###################################################################################   
    fig = plt.figure(figsize = (15 ,10))
    plt.xlabel(r'Distance to Root')
    plt.ylabel(r'Leaf Density')
    plt.title(r'Leaf Distribution')
    plt.xlim(min(heights) - .1 , max(heights)+1)
    plt.ylim(0 -.1,  max(leaf_freq) +.1)
    plt.errorbar((heights), (leaf_freq),  yerr= std)
    plt.plot((heights), (leaf_freq), 'o')
    filename_for_pdf = '/'+'HL.pdf'
    plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
    

def leaves_per_timestep(**kwargs):
    """
    This experiment performs two main tasks.  It plots:
    
    (1) timesteps vs. number of leaves.
        The relationship is linear and obtains m, b such that  y = mx+ b approximates y (leaves) and x (timesteps)
    (2) Collecting m's from (1), we plot arrival rate vs. rate of leaf addition
    
    Remark: Getting a readable (2) plot likely renders (1) plot to be less readable (too many lines)
    """
    #CREATING DIRECTORIES####################################################################
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
    dir_name1 = '/leaves_per_timestep' + date   
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    ######################################################################################
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 500000)
    number_of_experiments   = kwargs.pop('number_of_experiments', 100)
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    min_arrival_parameter   = kwargs.pop('min_arrival_parameter', 1)     # constant_arrval = 3
    max_arrival_parameter   = kwargs.pop('max_arrival_parameter', 3)
    arrival_parameter_step  = kwargs.pop('arrival_parameter_step', 1)
    preference              = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    #######################################################################################
    
    #CREATING ARRIVAL INTERVAL#############################################################
    """
    list goes from min_arrival_parameter to max_arrival_parameter
    """
    k = (max_arrival_parameter-min_arrival_parameter +1)/arrival_parameter_step
    arrival_interval = [0]*(k)
    for j in range(k):
        arrival_interval[j] = (min_arrival_parameter+j*arrival_parameter_step,)
    #######################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Leaf Count:\n\n'
    file.write(Info)    
    Info = 'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d \n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__name__ \
            +'Min Arrival Parameter: %3d\n'%min_arrival_parameter\
            +'Max Arrival Parameter: %3d\n'%max_arrival_parameter\
            +'Step Size: %3d\n'%arrival_parameter_step\
            +'Regarding leaf_count.pkl file:\n'\
            +'This returns a dictionary data with the following keys and values:\n'\
            +'\'arrival_list\'        ---> arrival_list (parameters of integers that denote how many nodes added at each timestep)\n'\
            +'\'slope_list\'          ---> list of slopes indexed by arrival list corresponding to change in number of leaves each timestep\n'\
            +'\'slope_of_slopes\'     ---> (m, b) (tuple) where m, b correspond to the slope of the arrival_list v slope_list curve \n'\
            +'arrival_parameter (int) ---> (x, y, m, b) (tuple) where x = timsteps, y = the number of leaves indexed by timesteps, \n'\
            +'                             m = approximate slope of the line, b = y-intercept'
    file.write(Info)
    file.close()
    #########################################################################################
    
    
    #THE EXPERIMENTS#########################################################################
    slope_list =[]
    arrival_list = []
    data = {}
    
    logger = open('log_leaves.txt', 'wb')
    logger.write('<<<<LEAF LOG>>>>\n\n')
    logger.write('Total Experiments for each arrival: %4d \n'%number_of_experiments)
    logger.write('###########################\n\n')
    logger.close()
    
    #ARRIVAL PARAMETER LOOP##########################################
    for arrival_parameter in arrival_interval:
        kwargs['arrival_parameters'] = arrival_parameter
        
        logger = open('log_leaves.txt', 'ab')
        logger.write('\n\nARRIVAL PARAMETER: %5d\n\n'%arrival_parameter[0])
        logger.close()
        ##########################################################
        #Arrival parameter is given by (k, )
        arrival_list.append(arrival_parameter[0])
        ##########################################################
         
        #INITIALIZE VARIABLES FOR FIXED ARGUMENTS#################
        number_of_leaves_per_timestep = []    #total number of leaves tally
        iteration_reached_per_experiment = [] #which timesteps were reached (not clear at outset when varying number of nodes)
        ##########################################################
        
        
        for k in range(number_of_experiments):
            
            logger = open('log_leaves.txt', 'ab')
            logger.write('EXP #: %5d\n'%k)
            logger.close()
            
            #PTREE GROWTH#############################################
            G = PTree(**kwargs)
            G.growth()
            ##########################################################
            
            #VARIABLES FROM PTREE#####################################
            exp_leave_list = G.get_number_of_leaves_per_timestep()
            n_glob = len(number_of_leaves_per_timestep)
            n_exp = len(exp_leave_list)
            ##########################################################
            
            ##########################################################
            """
            Though the number of timesteps shouldn't changed when no
            pruning is present, we make sure that if the number of time-steps
            is consistent over all experiments
            """
            if n_exp <= n_glob:
                exp_leave_list.extend([0]*(n_glob - n_exp))
                temp_experiment_list = [1]*n_exp +[0]*n_glob
            else:
                number_of_leaves_per_timestep.extend([0]*(n_exp - n_glob))
                iteration_reached_per_experiment.extend([0]*(n_exp - n_glob))
                temp_experiment_list = [1]*n_exp
            ##########################################################
            
            ##########################################################
            number_of_leaves_per_timestep[:] = [sum(x) for x in zip(number_of_leaves_per_timestep, exp_leave_list)]
            iteration_reached_per_experiment[:]   = [sum(x) for x in zip(iteration_reached_per_experiment, temp_experiment_list)]
            ##########################################################
        
        ##########################################################
        del number_of_leaves_per_timestep[0] #the first timestep always has a fixed number of nodes does not factor into growth
        del iteration_reached_per_experiment[0] #Same as above
        ##########################################################
        
        #COMPUTE AVERAGES#########################################
        y1_list  = np.array(number_of_leaves_per_timestep)
        y2_list  = np.array(iteration_reached_per_experiment)
        y        = y1_list/y2_list #Number of total leaves (frequency)/ number of experiments for which this is relevant
        ##########################################################
        
        #TIMESTEP AXIS############################################
        x        = np.arange(1, len(y)+1)
        ##########################################################
        
        #COMPUTE LINEAR APPROX AND RECORD SLOPE###################
        m, b = np.polyfit(x, y, 1)
        slope_list.append(m)
        ##########################################################
        
        #DATA DUMP################################################
        data[arrival_parameter[0]] = (x, y, m, b) #arrival parameter (k, )
        ##########################################################
    
    

    
    #LINEAR APPROX###############################################
    m, b = np.polyfit(arrival_list, slope_list, 1)
    ##########################################################
    
    #DATA DUMP################################################
    data['arrival_list'] = arrival_list
    data['slope_list']   = slope_list
    data['slope_of_slopes'] = (m, b)
    file = open('leaf_count.pkl', 'wb')
    pickle.dump(data, file)
    file.close()
    ##########################################################

#######################################################################################################################################################################

#NODE DESNITY########################################################################################################################################################
def node_density(**kwargs):
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 1000)
    number_of_experiments   = kwargs.pop('number_of_experiments', 4)
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters      = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference              = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Degree_Density' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Degree Density:\n\n'
    file.write(Info)    
    Info = 'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d\n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameters: %s\n\n'%str(arrival_parameters)\
            +'Regarding HL.pkl file:\n'\
            +'This returns a dictionary distance_to_root/height (key) to a list of the frequency of leaves indexed by experiment(values)'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #THE EXPERIMENT##########################################################################
    print 'Degree Density Experiment'
    print '##############################'
    DD_dict = {}
    file = open('DD.pkl', 'wb')
    for k in range(number_of_experiments):
        print 'EXP #: ', k
        G = PTree(**kwargs)
        G.growth()
        levels = G.get_levels()
        for height in range(len(levels)):
            number_of_nodes_temp= 0.0
            for node in levels[height]:
                number_of_nodes_temp +=1
            DD_dict.setdefault(height, []).append(number_of_nodes_temp/G.number_of_nodes())
                   
    ##########################################################################################
    
    
    #THE DUMP#################################################################################
    data = DD_dict
    pickle.dump(data, file)
    file.close()
    ########################################################################################
    
    #OPEN PICKLE FILE#######################################################################
    file = open('DD.pkl', 'rb')
    DD_dict = pickle.load(file)
    heights = np.array(sorted(DD_dict.keys()))
    DDen = np.array([np.mean(DD_dict[h]) for h in heights])
    std= np.array([np.std(DD_dict[h]) for h in heights])
    ########################################################################################
    
    #PLOT###################################################################################   
    fig = plt.figure(figsize = (15 ,10))
    plt.xlabel(r'Distance to Root')
    plt.ylabel(r'Degree Density (Average)')
    plt.title(r'Degree Density')
    plt.xlim(min(heights) - .1 , max(heights)+1)
    #plt.ylim(0 -.05,  1 +.05)
    plt.errorbar((heights), (DDen),  yerr= std)
    plt.plot((heights), (DDen), 'o')
    filename_for_pdf = '/'+'DD.pdf'
    plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
    

def node_density_node_peaks_experiment(**kwargs):
    """
    This computes the node density when a tree is conditioned to be a certain size.
    
    There are some subtleties in experimental design.  First off, we must estimate the denisty.  We use a standard histogram
    method.  That is to say, we run the tree growth 'number_of_experiments'-times.  Each growth yields some probability density whose
    input is height.  We average each of these densities together.  In theory certain, levels may not have a density; however we do not
    pad that with a zero.
    """
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes_interval     = kwargs.pop('max_allowable_nodes_interval', np.linspace(5000, 50000, 10))
    number_of_experiments            = kwargs.pop('number_of_experiments', 100)
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters               = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference                       = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    seed                             = kwargs.setdefault('seed' , (3, 3) )
    clustering                       = kwargs.pop('clustering', True)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/PEAK' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Degree Distribution:\n\n'
    file.write(Info)    
    Info = 'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes Interval: %4d %4d\n'%(max_allowable_nodes_interval[0], max_allowable_nodes_interval[-1])\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameters: %s\n\n'%str(arrival_parameters)\
            +'Regarding ND.pkl file:\n'\
            +'This returns a dictionary distance_to_root (or height) (key) to a list of average number of nodes indexed by each experiment(values)'
    file.write(Info)
    file.close()
    #############################################################################################
    
    #DATA INITIALIZATION#########################################################################
    data = {}
    #############################################################################################
    
    #LOG#########################################################################################
    if clustering:
        log = open('Node_Density_Log.txt', 'wb')
        log.write('<<<<<<Node Density Log >>>>>>>>>>\n')
        log.close()
    else:
        print 'NODE DENSITY EXPERIMENT'
        print '##############################'
    #############################################################################################
    
    #THE LOOP####################################################################################
    for max_allowable_nodes_pair in enumerate(max_allowable_nodes_interval):
        
        #LOGGING THE FIRST LOOP##################################################################
        if clustering:
            log = open('Node_Density_Log.txt', 'ab')
            log.write('\n\nMax Nodes: %5d\n'%max_allowable_nodes_pair[1])
            log.write('##############################\n\n')
            log.close()
        else:
            print '%s'%max_allowable_nodes_pair
        #########################################################################################
        
        #UPDATE MAX ALLOWABLE NODE PAIR##########################################################
        max_allowable_nodes_pair = max_allowable_nodes_pair[0], int(max_allowable_nodes_pair[1])  #replacing second component with int for safe usage in next line! and later
        kwargs['max_allowable_nodes'] = max_allowable_nodes_pair[1]                               #For PTree __init__
        #########################################################################################
        
        #THE EXPERIMENT##########################################################################
        ND_dict = {}       # the probability density when N = max_allowable_nodes
        ND_data_list = []  # the frequency of heights (this data will then be used to fit a gamma curve)
        file = open('ND.pkl', 'wb')
        for k in range(number_of_experiments):
            if clustering:
                log = open('Node_Density_Log.txt', 'ab')
                log.write('EXP #: %5d \n'%k)
                log.close()
            else:
                print 'Exp #:', k
            G = PTree(**kwargs)
            G.growth()
            levels = G.get_levels()
            for height in range(len(levels)):
                number_of_nodes_temp= 0.0
                for node in levels[height]:
                    ND_data_list.append(height)
                    number_of_nodes_temp +=1
                ND_dict.setdefault(height, []).append(number_of_nodes_temp/G.number_of_nodes())
        data[max_allowable_nodes_pair] = ND_dict, ND_data_list
        #######################################################################################
    
    ###########################################################################################
    """
    This is where we pad the ND_dict with zeros if the level didn't show up in a particular
    experiment.
    """
    for height in ND_dict.keys():
       if len(ND_dict[height])<number_of_experiments:
           k = number_of_experiments - len(ND_dict[height])
           ND_dict[height] += ([0]*k)
    ###########################################################################################
    
    #THE DUMP#################################################################################
    pickle.dump(data, file)
    file.close()
    ########################################################################################


def node_density_node_peaks_plot(**kwargs):
    """
    This plots the above experiment node_density_node_peaks
    """
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes_interval     = kwargs.pop('max_allowable_nodes_interval', np.linspace(6000, 6000, 1))
    number_of_experiments            = kwargs.pop('number_of_experiments', 1)
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters               = kwargs.setdefault('arrival_parameters', (4,))     # constant_arrval = 3
    preference                       = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    seed                             = kwargs.setdefault('seed' , (3, 3) )
    clustering                       = kwargs.pop('clustering', True)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/PEAK' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    ########################################################################################
    fig = plt.figure()#figsize = (15 ,10))
    file = open('ND.pkl', 'rb')
    data = pickle.load(file)
    ##########################################################################################
    for max_allowable_nodes_pair in enumerate(max_allowable_nodes_interval):
        max_allowable_nodes_pair = (max_allowable_nodes_pair[0], int(max_allowable_nodes_pair[1]))
        #OPEN PICKLE FILE#######################################################################
        ND_dict = data[max_allowable_nodes_pair][0]
        ND_data_list = data[max_allowable_nodes_pair][1]
        heights = np.array(sorted(ND_dict.keys()))
        DDen = np.array([np.mean(ND_dict[h]) for h in heights])
        std= np.array([np.std(ND_dict[h]) for h in heights])
        ########################################################################################

        #PLOT DATA##############################################################################   
        plt.xlabel(r'Distance to Root')
        plt.ylabel(r'Node Density')
        plt.title(r'Node Density')
        fit_alpha, fit_loc, fit_beta=ss.gamma.fit(ND_data_list)
        plt.plot((heights), (DDen), 'o')
        plt.errorbar((heights), (DDen), label = r'$N = %s$'%str(max_allowable_nodes_pair[1]))#yerr= std
        
        #PLOT GAMMA##############################################################################   
        fit_alpha, fit_loc, fit_beta=ss.gamma.fit(ND_data_list)
        constants = (fit_alpha, fit_beta, fit_loc)
        pdf_of_gamma = gamma.pdf(heights, fit_alpha, loc = fit_loc, scale = fit_beta)
        plt.plot(heights, pdf_of_gamma, '.-.',label =r'$\sim\gamma(%2.1f, %0.2f, %2.1f)$'%constants )
        ##########################################################################################
        
    plt.legend()
    filename_for_pdf = '/'+'NodeDensity.pdf'
    plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
    #######################################################################################
    
    
def node_density_seed_peaks(**kwargs):
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 5000)
    number_of_experiments   = kwargs.pop('number_of_experiments', 4)
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters      = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference              = kwargs.setdefault('preference', height_preference)   #1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/PEAK' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Degree Distribution:\n\n'
    file.write(Info)    
    Info = 'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes Interval: %4d %4d\n'%(max_allowable_nodes_interval[0], max_allowable_nodes_interval[-1])\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameters: %s\n\n'%str(arrival_parameters)\
            +'Regarding HD.pkl file:\n'\
            +'This returns a dictionary distance_to_root/height (key) to a list of average degrees indexed by each experiment(values)'
    file.write(Info)
    file.close()
    #########################################################################################
    data = {}
    for max_allowable_nodes_pair in enumerate(max_allowable_nodes_interval):
        max_allowable_nodes_pair = max_allowable_nodes_pair[0], int(max_allowable_nodes_pair[1])
        kwargs['max_allowable_nodes'] = max_allowable_nodes_pair[1] #For PTree __init__
        #THE EXPERIMENT##########################################################################
        print 'Height V Degree Experiment'
        print '##############################'
        DD_dict = {}
        degree_data_list = []
        file = open('DD.pkl', 'wb')
        for k in range(number_of_experiments):
            print 'EXP #: ', k
            G = PTree(**kwargs)
            G.growth()
            levels = G.get_levels()
            for height in range(len(levels)):
                number_of_nodes_temp= 0.0
                for node in levels[height]:
                    degree_data_list.append(height)
                    number_of_nodes_temp +=1
                DD_dict.setdefault(height, []).append(number_of_nodes_temp/G.number_of_nodes())
        
        data[max_allowable_nodes_pair] = DD_dict, degree_data_list
    
    #THE DUMP#################################################################################
    pickle.dump(data, file)
    file.close()
    ########################################################################################
    fig = plt.figure(figsize = (15 ,10))
    file = open('DD.pkl', 'rb')
    data = pickle.load(file)
    ##########################################################################################
    for max_allowable_nodes_pair in enumerate(max_allowable_nodes_interval):
        max_allowable_nodes_pair = (max_allowable_nodes_pair[0], int(max_allowable_nodes_pair[1]))
        #OPEN PICKLE FILE#######################################################################
        DD_dict = data[max_allowable_nodes_pair][0]
        heights = np.array(sorted(DD_dict.keys()))
        DDen = np.array([np.mean(DD_dict[h]) for h in heights])
        std= np.array([np.std(DD_dict[h]) for h in heights])
        ########################################################################################

        #PLOT###################################################################################   
        plt.xlabel(r'Distance to Root')
        plt.ylabel(r'Degree Density (Average)')
        plt.title(r'Degree Density')
        #plt.xlim(min(heights) - .1 , max(heights)+1)
        plt.plot((heights), (DDen), 'o')
        plt.errorbar((heights), (DDen), label = str(max_allowable_nodes_pair[1]))#yerr= std
        
    plt.legend()
    filename_for_pdf = '/'+'NodeDensity.pdf'
    plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
    #######################################################################################

#######################################################################################################################################################################

#WEIGHT################################################################################################################################################################
def weight_watcher(**kwargs):
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 2000)
    number_of_experiments   = kwargs.pop('number_of_experiments', 4)
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters      = kwargs.setdefault('arrival_parameters', (100,))     # constant_arrval = 3
    preference              = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    #######################################################################################
    
    
    G = PTree(**kwargs)
    G.growth()
    y = G.weight_tracker
    x = np.arange(1, len(y)+1)
    plt.plot(x, y, label= 'Total Weight')
    m, b = np.polyfit(x, y, 1)
    z = m*x + b
    plt.plot(x, z, label = r'$y = %1.2fx + %1.2f$'%(m, b))
    plt.xlabel('Time Step')
    plt.ylabel('Total Weight')
    plt.legend()
    plt.show()
    
def leaf_watcher(**kwargs):
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 100)
    number_of_experiments   = kwargs.pop('number_of_experiments', 1)
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters      = kwargs.setdefault('arrival_parameters', (1,))     # constant_arrval = 3
    preference              = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/LeafWatching' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    G = PTree(**kwargs)
    G.growth()
    y = G.the_leaf_distance
    x = np.arange(0, len(y))
    file = open('leaf_watcher.pkl', 'wb')
    pickle.dump([x, y], file)
    file.close()
    #plt.ylim(0, max(y)+.5)
    #plt.plot(x, y)
    #plt.xlabel('Time Step')
    #plt.ylabel('Total Distnace')
    #plt.show()
    
def leaf_watcher_with_arrival_increasing(**kwargs):
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 10000)
    number_of_experiments   = kwargs.pop('number_of_experiments', 1)
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    preference              = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    arrival_rate_interval   = kwargs.pop('max_arrival_rate', [1]+range(10, 100, 10))   #1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/LeafWatching' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    data = {}
    for k in arrival_rate_interval:
        kwargs['arrival_parameters']= (k,) 
        G = PTree(**kwargs)
        G.growth()
        y = np.array(G.the_leaf_distance)
        x = np.arange(0, len(y))
        print y
        data[k] = x, y
        file = open('leaf_watcher_arrival_increasing.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
     
#######################################################################################################################################################################



#######################################################################################################################################################################
if __name__== '__main__':
    leaf_watcher_with_arrival_increasing()
    #leaves_per_timestep()
    #leaf_watcher()
    #weight_watcher()   