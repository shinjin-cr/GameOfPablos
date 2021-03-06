import numpy as np
#import matplotlib.pyplot as plt
import os
from random import choice 
from numpy.random import binomial
import datetime
import random
import networkx as nx
from numpy.random import poisson as pois
import types
import cProfile

##########################################################################################################################################################################
def constant_arrival(*args):
    """
    Returns a function that is constantly args[0]
    """
    f = lambda node : args[0]
    return f

def no_more_leaves(self, node):
    """1/(distance_to_closest_leaf+1)"""
    return 1./(self.node[node]['distance_to_closest_leaf']+1.)

def height_preference(self, node):
    """ weight = height + 1"""
    return self.node[node]['height']+1.
def just_leaves(self, node):
    return int(node in self.leaves)
##########################################################################################################################################################################

##########################################################################################################################################################################
class PTree(nx.DiGraph):
    """
    (P)referential Attachment (Tree) is an object that is a recursively growing tree.  We add a fixed number of nodes 
    at each time step with the probability of attaching to an existing node proportional to a preference function 
    defined in the __init__().  It uses a DiGraph() object from the NetworkX Library.
    """
    def __init__(self, arrival_distribution = constant_arrival, arrival_parameters = (3,), max_allowable_nodes = 10, preference = None, seed = None):
        """
        The initialized tree begins with a single root node 0.  We can initialize the tree to be entire syndicate as well specified by seed, which
        we assume is complete and of uniform degree.
        --------------------------------------------------------------------------
        Input:
            self
            arrival distribution (int)     Probability distribution of new nodes
                                           added.  For our purposes, this will
                                           be the constant distribution.
            arrival_parameters (tuple)     These are the parameters that are rele-
                                           vant for the arrival distribution.  In
                                           the case of a constant distribution, this
                                           is just the constant.
            max_allowable_nodes (int)      This is the total number of nodes that 
                                           the network is permitted to grow to.
            preference (function)          Function : Nodes (self) ----> Real #'s
                                           This function determines how much each
                                           preference is given when attaching nodes.
            seed (int, int)                The (height, degree) of the initial seed
                 (height, degree)          where we initialize the tree to be a complete
                                           k-ary tree of a specified height, where k =
                                           degree
        --------------------------------------------------------------------------
        Output:
            None
        --------------------------------------------------------------------------
        Description:
        
            We mention the relevant immutable traits (can be changed, but should not be):
                
                self.arrival_parameters (tuple)                   = (k,); usually constant for distribution below
                                                                    tuple format allows for built in probability distributions in scipy
                self.arrival_distribution (function)              = this is just a constant function for the relevant model; earlier models
                                                                    permitted this to be a Poisson or even Binomial distribution
                self.max_allowable_nodes (int)                    = stop growth at this integer as we assume this goes to infinity
                self.max_nodes_reached (bool)                     = True or False whether total number of nodes currently exceeds 
                                                                    max allowable nodes
                self.prefenerence (function)                      = see preference comments below
            
            We mention the relevant (mutable) traits--here by mutable we don't mean those changed by the user (don't do it!) more as 
            in the traits change over time as the network grows or shrinks.  The traits of the graph in general should be used by the
            "getter" functions:
            
                self.node[node]['height']  (int)                  = depth of node/number of edges between node and root
                self.node[node]['distance_to_closest_leaf'] (int) = minimum number of edges between node and descendent leaf
                self.node[node]['shortest_path_to_root'] (list)   = list beginning at node and ending at the root (0)--inclusive!
                self.timestep (int)                               = each time the tree is altered we adjust timestep (for
                                                                    visualization)
                self.the_leaf_distance (list)                     = the list of maximum distances to the leaf (max of 
                                                                    self.node[node]['distance_to_closest_leaf'] for each node)
                self.weight_tracker (list)                        = total weight of the network
                
            Internal Traits are the ones used for book-keeping within the object itself:
                
                self.leaves_removed (list)                        = these are the current leaves that are removed
                self.node_count (int)                             = should be thought of as node counter; this is the total number of nodes
                                                                    that have been in the network for its entire life
        """
        nx.DiGraph.__init__(self)
        self.add_node(0)
        self.root = 0
        
        #GRAPH DEPENDENT TRAITS###################################
        self.node[0]['height'] = 0 
        self.node[0]['distance_to_closest_leaf'] = 0
        self.levels = [[0]] 
        self.leaves = [0]
        self.new_nodes = [0]
        self.nodes_added = [1]
        self.node_count = 1
        self.timestep = 0
        self.weights = []
        self.leaves_removed = []
        self.number_of_leaves_per_timestep = [1]
        self.weight_tracker = [1]
        self.the_leaf_distance = [0]
        self.node[0]['shortest_path_to_root'] = [0]
        #########################################################
        
        #GRAPH HEIGHT###################################
        #self.max_height = 0
        #self.min_height = 0
        #self.avg_height = 0
        #########################################################
        
        #GRAPH INDEPENDENT TRAITS################################
        self.arrival_parameters = arrival_parameters
        self.arrival_distribution = arrival_distribution(*arrival_parameters)
        self.max_allowable_nodes = max_allowable_nodes
        self.max_nodes_reached = False
        #########################################################
        
        #PREFERENCE##############################################
        """
        --------------------------------------------------------------------------
        Input:
            self
            node (int)                  We are labeling nodes by integers
        --------------------------------------------------------------------------
        Output:
            weight (float)              Each node is added to the graph with 
                                        probability proportional to its weight in
                                        in the network.  This weight is computed
                                        from node attributes
        --------------------------------------------------------------------------
        Description:
            The weight here is a function of height alone.
        """
        if preference == None:
            preference = no_more_leaves
        self.preference = types.MethodType(preference, self)
        #########################################################
        
        #SEED####################################################
        if seed != None:
            height, degree = seed
            if degree ==1:
                self.nodes_added = [height]
            else:
                self.nodes_added = [(degree**(height+1) - 1)/(degree - 1)]
            
            #GRAPH HEIGHT###################################################
            #self.max_height, self.min_height, self.avg_height  = (height,)*3
            ################################################################
            
            self.node[0]['distance_to_closest_leaf'] = height
            for k in range(height):
                nodes_to_update = self.levels[-1]
                level_temp = []
                dist_to_closest_leaf_temp = height - k -1
                for new_parent in nodes_to_update:
                    for j in range(degree):
                        child = self.node_count
                        self.add_edge(new_parent, child)
                        level_temp.append(child)
                        self.node[child]['height'] = k+1
                        self.node[child]['shortest_path_to_root'] = [child]+self.node[new_parent]['shortest_path_to_root']
                        self.node[child]['distance_to_closest_leaf'] = dist_to_closest_leaf_temp
                        self.node_count +=1
                self.levels.append(level_temp)
            self.leaves = list(self.levels[-1])
            self.the_leaf_distance[0] = height    
            
         
        
        #TOTAL WEIGHT!################################################    
        self.weights = [self.preference(node1) for node1 in self.nodes()]
        self.total_weight = sum(self.weights)
        self.weight_tracker[0] = self.total_weight
        #############################################################  
                
        
        #########################################################
    
    
    #THE GROWTH PROCESS#################################################################################
    def arrival(self):
        """
        --------------------------------------------------------------------------
        Input:
            self
        --------------------------------------------------------------------------
        Output:
            number_of_new_nodes (int)           New Nodes added at each time step
        --------------------------------------------------------------------------
        Description:
            This the arrival process for which the total number of nodes added to
            the network is added.  The distribution and it's parameters are both
            specified by the user.  The default is a Poisson distribution.
        """
        number_of_new_nodes = int(self.arrival_distribution(self.arrival_parameters))
        return number_of_new_nodes
    
    def node_from_preference(self):
        """
        --------------------------------------------------------------------------
        Input:
            self
            node (int)                  Labeling nodes with integers
        --------------------------------------------------------------------------
        Output:
            node (int)                  The node that is randomly chosen according
                                        to the probability of preference
        --------------------------------------------------------------------------
        Description:
            This algorithm follows the Roulette Wheel Selection process frequently
            used in genetic modeling.  The preference() function assigns weights
            to each node and then we select a node accordingly. 
        """
        nodes = self.nodes()
        total_weight= self.total_weight
        arrow = np.random.uniform(0, total_weight)
        temp_sum = 0
        for index in range(self.number_of_nodes()):
            temp_sum += self.weights[index]
            if temp_sum > arrow:
                return nodes[index]
        return int(nodes[index])
    
    def add_nodes(self): 
        """
        --------------------------------------------------------------------------
        Input:
            self
        --------------------------------------------------------------------------
        Output:
            None        
        --------------------------------------------------------------------------
        Description:
            This function updates the PTree and the node attributes.  We perform
            the following process:
                (1) Nodes arrive according to self.arrival()
                (2) New Nodes Assigned a Parent according to self.preference()
                (3) PTree is updated
        It is important that new nodes DO NOT add new nodes of their own
        until the next time step!
        """
        
        
        
        #ARRIVAL################################################################################################
        number_of_new_nodes = self.arrival()
        self.nodes_added.append(number_of_new_nodes)
        self.new_nodes = []   #For Visualization
        ########################################################################################################
        
        #NEW NODES ASSIGNED PARENT##############################################################################
        edges_to_add = []
        
        for k in range(number_of_new_nodes):
            """
            We first find (parent, child) links/edges
            and then update the tree afterwards
            """
            parent = self.node_from_preference()
            child = int(self.node_count)
            edges_to_add.append((parent, child))
            self.node_count +=1
            self.new_nodes.append(child)  #For Visualization
        ####################################################
        
        #UPDATE TREE############################################################################################
        
        #TRAITS##############################################
        self.add_edges_from(edges_to_add)
        for parent, child in edges_to_add:
            try:
                self.leaves.remove(parent)
            except ValueError:
                pass
            self.leaves.append(child)
            self.node[child]['height'] = self.node[parent]['height'] +1
            child_height = self.node[child]['height']
            try:
                self.levels[child_height].append(child)
            except IndexError:
                self.levels.append([])
                self.levels[child_height].append(child)
            
            #PATHS##################################################
            self.node[child]['shortest_path_to_root'] = [child] + self.node[parent]['shortest_path_to_root']
            self.node[child]['distance_to_closest_leaf'] = 0
            ########################################################
            
            #HEIGHTS#######################################################
            #heights = nx.get_node_attributes(self, 'height').values()
            #heights.remove(0) #Pablo's Height
            #self.max_height = max(heights)
            #self.min_height = min(heights)
            #self.avg_height = np.mean(heights)
            ##################################################################
            
        ####################################################
        
        #DISTANCE TO LEAF###################################
        """
        We take leaves of ordered by height in decreasing order and then 
        update the distance to the leaves using a breadth-first search.
        """ 
        new_nodes_sorted_by_height = sorted(self.new_nodes, key = lambda node: self.node[node]['height'])
        traversal_dictionary = {node: False for node in self.nodes()}########################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for new_node in new_nodes_sorted_by_height:
            for node_to_update in self.node[new_node]['shortest_path_to_root']:
                if traversal_dictionary[node_to_update]:
                    break
                a_near_leaf = self.bfs_searching_for_near_leaf(node_to_update)
                self.node[node_to_update]['distance_to_closest_leaf'] = self.node[a_near_leaf]['height'] - self.node[node_to_update]['height']
                traversal_dictionary[node_to_update] = True
        ####################################################
        
        #NUMBER OF LEAVES AFTER ARRIVAL AND LEAVES ADDED########################################################
        self.number_of_leaves_per_timestep.append(len(self.leaves))
        ########################################################################################################
        
        #WEIGHT TRACKER#####################################
        self.weights = [self.preference(node1) for node1 in self.nodes()]
        self.total_weight= sum(self.weights)
        self.weight_tracker.append(self.total_weight)
        ####################################################
        
        #The Leaf Distance Tracker#####################################
        self.the_leaf_distance.append(max(nx.get_node_attributes(self, 'distance_to_closest_leaf').values()))
        ########################################################################################################
        
        ########################################################################################################            
    
    def growth(self):
        """
        --------------------------------------------------------------------------
        Input:
            self
        --------------------------------------------------------------------------
        Output:
            None
        --------------------------------------------------------------------------
        Description:
           This grows the network under the rules specified until the tree reaches
           its maximum_allowable_number_of_nodes.
        """
        #########################################################
        #print 'Max # of Nodes: ', self.max_allowable_nodes
        #########################################################
        while(self.number_of_nodes() < self.max_allowable_nodes ):
            self.add_nodes()
    def prune(self, node_to_remove):
        """
        --------------------------------------------------------------------------
        Input:
            self
            node_to_remove (int)   the node on the graph that will be removed
        --------------------------------------------------------------------------
        Output:
            None
        --------------------------------------------------------------------------
        Description:
           Removes node_to_remove and all of its descendants.  Also, updates
           relevant traits of the tree.
        """
        #GET NODES THAT WILL BE REMOVED#################################
        nodes_removed = list(nx.shortest_path_length(self, source=node_to_remove).keys())
        nodes_removed.sort(key = lambda node: self.node[node]['height'])  #Sorting helpful when updating levels
        ################################################################
        
        ################################################################
        self.leaves_removed = []
        ################################################################
        
        #UPDATE TRAITS##################################################
        
        #UPDATE HEIGHT, LEVELS, AND LEAVES (PT. 1)#######################
        """
        self.leaves updated in the following way: if a node was a leaf
        and is about to be removed it is removed from self.leaves.
        
        The other update required is the nodes that weren't leaves
        on the previous timestep and now are.  This is done later.
        
        Also, our level updates could leave empty lists [].  This
        is taken care of later as well.
        """
        
        for node in nodes_removed:
            height = self.node[node]['height']
            self.levels[height].remove(node)
            try:
                index = self.leaves.index(node)
                leaf = self.leaves[index]
                self.leaves_removed.append(leaf)
                del self.leaves[index]
            except ValueError:
                pass
        
        ###############################################################
        
        ###############################################################
        """ 
        Remove empty list [] from levels.
        """
        self.levels = [level for level in self.levels if level != []]
        ###############################################################
        
        ###############################################################
        """
        Remove nodes and keep track of parent of the subtree removed.
        """
        parent = self.predecessors(node_to_remove)[0]
        self.remove_nodes_from(nodes_removed)
        ###############################################################
        
        #UPDATE LEAVES (PT. 2) AND DISTANCE TO LEAVES#############################
        """
        We have to update the parent of the subtree, namely if the
        root of the subtree was the parent's only child, then he
        now is a leaf.  Stated another way, if the parent's out
        degree == 1 before the removal, then he is now a leaf.
        """
        #UPDATE PARENT OF REMOVED NODE FIRST##################
        if self.out_degree(parent) == 0:
            self.leaves.append(parent)
            self.node[parent]['distance_to_closest_leaf'] = 0
        else:
            self.node[parent]['distance_to_closest_leaf'] = \
            min([(self.node[child]['distance_to_closest_leaf'] +1) for child in self.successors(parent)])
        #######################################################
        
        #UPDATE ALL OTHER NODES FROM PARENT TO ROOT############
        if parent == 0:
            pass
        else:
            path = list(self.node[parent]['shortest_path_to_root'])
            del path[0] #delete the parent since we already updated it
            for node in path:
                self.node[node]['distance_to_closest_leaf'] = \
                min([(self.node[child]['distance_to_closest_leaf'] +1) for child in self.successors(node)])
        #######################################################
                
        ##############################################################################
        
        #UPDATE GLOBAL HEIGHTS##########################################
        """
        Each node has a height, but so does the tree.  This is updated
        here.
        """
        #heights = nx.get_node_attributes(self, 'height').values()
        #heights.remove(0) #Pablo's Height
        #if heights == []:
        #    self.max_height, self.min_height, self.avg_height = (0, 0, 0)
        #else:
        #    self.max_height = max(heights)
        #    self.min_height = min(heights)
        #    self.avg_height = np.mean(heights)
        ###############################################################
        
        ###############################################################
        self.weights = [self.preference(node1) for node1 in self.nodes()]
        ###############################################################
        
    ####################################################################################################
    
    #ALGORITHMS########################################################################################
    def bfs_searching_for_near_leaf(self, node):
        """
        --------------------------------------------------------------------------
        Input:
            self
            node (int)   the node on the graph that begin our search
        --------------------------------------------------------------------------
        Output:
            child (int)  the child of the node that represents that is one of the nodes
                         of minimum distance
        --------------------------------------------------------------------------
        Description:
           By doing a breadth first search, we look for a leaf closest to the node inputted.
        """
        if self.out_degree(node)==0:
            return node
        nodes_to_traverse = [node]
        for parent in nodes_to_traverse:    
            for child in self.successors(parent):
                if self.out_degree(child) == 0:
                    return child
                else:
                    nodes_to_traverse.append(child)
    
    ####################################################################################################
    
    #VISUALIZATION######################################################################################
    def visual(self, PU = None): 
        """
        IMPORTANT: the zen of visualization--call visual, then adjust time-step
        externally.  For instance,
        
        >> G.Ptree()
        >> G.visual()
        >> G.do_stuff_that_changes_G()
        >> G.timestep += 1
        
        --------------------------------------------------------------------------
        Input:
            self
            PU (PoliceUnit)       This is a pointer to the object created below.
                                  The default is None.  The function then accesses
                                  the attributes of the PoliceUnit object.
        --------------------------------------------------------------------------
        Output:
            Pdfs                  Saved to Directory "Data" in the same file as script        
        --------------------------------------------------------------------------
        Description:
            This uses the networkx draw() function to draw the graph and dots
            library for making hierarchal structure clear. 
        
        """
        
        #############################################################################################################################################
        print self.timestep
        #############################################################################################################################################
        
        #Lists that are collected from PU object###################################################################################################### 
        investigated                              = []
        list_of_nodes_occupied_by_officer         = []
        removed_nodes                             = []
        #############################################################################################################################################
        
        #############################################################################################################################################
        if PU == None:
            pass
        else:
            removed_nodes[:]                      = PU.removed_nodes
            list_of_nodes_occupied_by_officer[:]  = [officer1.get_current_node() for officer1 in PU.officers]
            investigated                          = PU.investigated    
        #############################################################################################################################################        
        
        #COLORS######################################################################################################################################
        fig = plt.figure()
        plt.rc('font',**{'family':'serif'})
        color_dictionary = {node: (      '#CC0000'   if node in removed_nodes
                                    else '#FF66FF'   if node in list_of_nodes_occupied_by_officer
                                    else '#99FFFF'   if node in investigated
                                    else '#CCFF66'   if node in self.new_nodes
                                    else '#00CC66') 
                                    for node in self.nodes()
                            }
        colors = [color_dictionary.get(node, '#FFFFFF') for node in self.nodes()]
        #############################################################################################################################################
        
        
        #WEIGHTS#####################################################################################################################################
        self.weights = [self.preference(node) for node in self.nodes()]
        total_weight= sum(self.weights)
        #############################################################################################################################################
        
        #LABELS#######################################################################################################################################
        label_dictionary = { node: \
                                   ''
                                   +'%s\n'%str(self.node[node]['shortest_path_to_root'])\
                                   +r'$v_{%s}$'%str(node) + '\n'\
                                   #+r'$w_{%s} = %.5f$'%(self.timestep , self.weights[k]) \
                                   #+'%4d'%self.node[node]['distance_to_closest_leaf']
                                   #+r'$\mathbb{P}_{%s} = %.5f$'%(self.timestep +1, self.weights[node]/total_weight)
                          for k, node in enumerate(self.nodes())
                          } #if node !=0 else 'Eve' v_{%s}$
        #############################################################################################################################################
        
        
        #SHAPES#######################################################################################################################################
        shape_dictionary = {node: (      'p'   if node in removed_nodes
                                    else '>'         if node in list_of_nodes_occupied_by_officer
                                    else '<'         if node in investigated
                                    else 's'         if node in self.new_nodes
                                    else 'o') 
                                    for node in self.nodes()
                            }
        shapes = [shape_dictionary.get(node, 'o') for node in self.nodes()]
        #############################################################################################################################################
        
        #SIZE########################################################################################################################################
        node_sizes_list = [200*3 if node!=0 else 400*3 for node in self.nodes()] #Everthing Else
        #node_sizes_list = [10000 if node!=0 else 10000 for node in self.nodes()] #FOR PAPER WRITEUPS
        #############################################################################################################################################
        
        
        #DRAW########################################################################################################################################
        pos=nx.graphviz_layout(self,prog='dot')
        nx.draw(self,
                pos,
                
                #labels##################
                with_labels=True,
                labels = label_dictionary,
                ###########################
                
                arrows=False, 
                node_color = colors,
                font_family = 'serif',
                font_size = 8,
                node_size = node_sizes_list,
                #node_shape = shapes
                )
        #############################################################################################################################################
        
        #CHANGING DIRECTORIES########################################################################################################################
        """
        We create two directories Data > EvolutionPics + /date/ within the same 
        file as the python script.  We then can save all plots, 
        data, and text to this directory.  Each timestep is order by an 
        integer in increasing order.
        """
        s = os.path.dirname(os.path.realpath(__file__))
        os.chdir(s)
        dir_name = '/Data'    
        if not os.path.exists(os.getcwd()+dir_name):
            os.makedirs(os.getcwd()+dir_name)
        os.chdir(os.getcwd() + dir_name)
        date =  str(datetime.datetime.now().strftime(" %m:%d:%y %H%M"))
        dir_name2 = '/VisualEvolution!'#'/EvolutionPics' + date    
        if not os.path.exists(os.getcwd()+dir_name2):
            os.makedirs(os.getcwd()+dir_name2)
        os.chdir(os.getcwd() + dir_name2)
        ##################################################################################################################################################
        
        ##################################################################################################################################################
        filename_for_pdf = '/'+ str(self.timestep)+'.pdf'
        plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
        ##################################################################################################################################################
    
    def seed_visual(self):
        """
        Visualize the seed of the process
        """
        print 'Seedling'
        self.visual()
        self.timestep+=1
    def add_nodes_with_visuals(self):
        """
        Visualize the Node Addition
        """
        print 'Adding'
        self.add_nodes()
        self.visual()
        self.timestep +=1
        
    def growth_with_visuals(self):
        """
        Visualize the growth
        """
        while(self.number_of_nodes() < self.max_allowable_nodes ):
            self.add_nodes_with_visuals()
    ####################################################################################################
    
    #GETTERS############################################################################################
    def get_levels(self):
        """
        Output: copy of self.levels
        """
        return list(self.levels)
    def get_max_allowable_nodes(self):
        """
        Output: copy of (int) self.max_allowable_nodes
        """
        return int(self.max_allowable_nodes)
    def get_leaves(self):
        """
        Output: copy of (list) self.leaves
        """
        return list(self.leaves)
    def get_number_of_leaves(self):
        """
        Output: the # of leaves in the network
        """
        return len(self.leaves)
    def get_number_of_leaves_per_timestep(self):
        return list(self.number_of_leaves_per_timestep)
    ####################################################################################################
    
    #DEBUGGING##########################################################################################
    def print_attributes(self):
        """
        Prints object attributes
        """
        print '############'
        print 'TIMESTEP: ',self.timestep
        print 'Nodes: ', self.nodes()
        print 'Heights:', nx.get_node_attributes(self, 'height').values()
        print 'Weights: ', self.weights
        print 'Levels: ',self.levels
        print 'Leaves: ', self.leaves
        print 'Number of Nodes Added: ', self.nodes_added
        print 'New Nodes: ', self.new_nodes
        print 'Node Count: ',self.node_count
        
        #HEIGHTS############################################
        #print 'Max Height: ', self.max_height
        #print 'Min Height: ', self.min_height
        #print 'Avg Height: ',self.avg_height
        ####################################################
        
        print '############'
        
    
    
    ####################################################################################################

##########################################################################################################################################################################    

##########################################################################################################################################################################
class Officer(object):
    """
    The Officer class is the object that performs investigation/arrests on the ptree.
    """
    
    def __init__(self, ptree, leaf):
        """
        --------------------------------------------------------------------------
        Input:
            self
            ptree (PTree)           The network the officer is investigating
            leaf (int)              The node on the network the officer begins
                                    investigating
        --------------------------------------------------------------------------
        Output:
            None
        --------------------------------------------------------------------------
        Description:
            The officer begins on a leaf and then either moves or arrests.  The
            strategy is specified not here, but in the PoliceUnit.  Relevant
            attributes:
                self.path (list)              = The nodes the police traverse.  This
                                                is relevant since we assume the
                                              = officers random walk is self-avoiding.
                self.current_node (int)         The current node the police officer
                                                is at.
                self.ptree (PTree)              The network the officer investigates
                self.hope_of_reaching_root    = Assuming the random walk the officer
                (bool)                          makes is self-avoiding the moment he
                                                moves away from the parent of the
                                                current node, he is doomed to fail.
                self.investigation_must_end   = When the investigation reaches another
                (bool)                          leaf.  That is to say, the investigation
                                                has reached its conclusion.  We will
                                                try not to run a random walk to its conclusion
                                                but sometimes it will be necessary.
        """
        self.path = [leaf]
        self.current_node = leaf
        self.ptree = ptree
        self.hope_of_reaching_root = True
        self.investigation_must_end = False

    #ACTIONS##########################################################################
    def investigate(self):
        """
        --------------------------------------------------------------------------
        Input:
            self
        --------------------------------------------------------------------------
        Output:
            None
        --------------------------------------------------------------------------
        Description:
            Takes all the nieghbors of the officers current node and randomly
            selects one to move.  If the officer moves in the direction of a child
            self.hope_of_reaching_root = False as her walk is self-avoiding.
        """
        #############################################################
        parent_list = self.ptree.predecessors(self.current_node)
        children_list = self.ptree.successors(self.current_node)
        #############################################################
        
        #############################################################
        parent = parent_list[0]                  #tree has unique parent
        neighbors = children_list + parent_list  #neighbors are parents and children
        if len(self.path) >= 2:                  #if we have made at least one move
            neighbors.remove(self.path[-2])      #remove the last node visited from neighbors
        if len(neighbors) == 0:                  #after removal if neibor set is non-empty continue
            self.investigation_must_end = True
            return 
        #############################################################
        
        #############################################################
        next_node = random.choice(neighbors)
        if next_node != parent:
            self.hope_of_reaching_root = False  #walk is self-avoiding! and tree has only one path to root
        self.path.append(next_node)
        self.current_node =  next_node
        #############################################################
        
    def arrest(self):
        """
        The officer arrests the current node and all of its descendents.
        """
        self.ptree.prune(self.current_node)
    ##################################################################################
    
    #GETTERS##########################################################################
    def get_current_node(self):
        """
        Output: returns a copy of self.current_node (int)
        """
        return int(self.current_node)
    def get_path(self):
        """
        Output: returns a copy of self.path (list)
        """
        return list(self.path)
    def get_hope(self):
        """
        Output: returns a copy of self.hope_of_reaching_root (bool)
        """
        return bool(self.hope_of_reaching_root)
    def get_nodes_arrested(self):
        """
        Output: returns a copy of the nodes that will be arrested (list)
        """
        return list(nx.shortest_path_length(self.ptree, source=self.current_node).keys())
    
    ##################################################################################
    
    #PRINT############################################################################
    def __str__(self):
        """
        Current Node: self.current_node (int)
        Hope: self.hope_of_reaching_root
        """
        return 'Current Node: ' + str(self.get_current_node()) +'\n' + 'Hope: '+ str(self.get_hope())
    ##################################################################################
    
##########################################################################################################################################################################

##########################################################################################################################################################################    
class PoliceUnit(object):
    """
    This is a collection of officers for a given time-step.  We initialize
    them based on choosing some fixed subset of the leaves.  Their goal is to eliminate
    the (criminal) network they are associated to.
    
    There is a strategy.txt  in the same directory as this .py file that  will describe 
    the strategies implemented here.   They are indexed by integers.
    
    In terms of the current model, we really only need one officer.  However, this object
    controls the strategy the officer employs.  This is the most important aspect for pur-
    suit experiments.
    
    The Strategies.txt describes the strategies implemented here in full and have 
    a full dictionary between the variables used here and the paper.
    """
    def __init__(self, ptree, strategy = 0, degree_threshold = None, officers_sent_out = None, number_of_investigations_for_strategy_SI = None, cost_experiment = False):
        """
        --------------------------------------------------------------------------
        Input:
            self
            ptree (PTree)           The network the PU investigates
            strategy (int)          The strategy (elaborated in the text file) that
                                    will be pursed by the unit.
        --------------------------------------------------------------------------
        Output:
            None
        --------------------------------------------------------------------------
        Description:
            This should be thought of as a collection of officers that organize
            themselves to eliminate the ptree network.  Here are the relevant
            attributes:
                
                self.officers (list of officers)    = list of officers currently investigating
                self.strategy (int)                 = strategy indexed by integer
                self.ptree (PTree)                  = the criminal network being inspected
                self.removed_nodes (list of ints)   = list of nodes that will be removed
                self.investigated (list of ints)    = list of nodes that are being investigated
                                                      by the officer that is CURRENTLY walking/
                                                      arresting
                self.root_found (bool)              = Boolean value when one officer has reached
                                                      the root
                self.officers_sent_out (int)        = The number of officers that are sent out;
                                                      again for the relevant model, this is 
                                                      simply 1.
                self.cost_experiment (bool)         = Normally, when we perform a pursuit with strategy 1
                                                      (only investigations/S_I), we do not need to continue
                                                      investigating if the officer takes a walk away from the root
                                                      instead of towards it.  However, when analyzing the total 
                                                      number of investigations (aka cost), we have to include
                                                      this because the officer in the model, does not 
                                                      know the global structure and will be doomed to investigate
                                                      until he cannot any longer.
         """
        self.officers = []
        self.strategy = strategy
        if degree_threshold == None:
            self.degree_threshold = 4
        else:
            self.degree_threshold = degree_threshold
        if number_of_investigations_for_strategy_SI == None:
            self.number_of_investigations_for_strategy_SI = 1
        else:
            self.number_of_investigations_for_strategy_SI = number_of_investigations_for_strategy_SI 
        self.ptree = ptree
        self.removed_nodes = []
        self.investigated = []
        self.total_investigations = 0
        self.total_arrests  = 0
        self.root_found = False
        if officers_sent_out == None:
            self.officers_sent_out = 1 #self.ptree.arrival_parameters[0] - 1
        else:
            self.officers_sent_out = officers_sent_out
        self.cost_experiment = cost_experiment
    
    
    
    #UPDATE###########################################################################
    def remove_extra_officers_before_arrest(self, officer):
        """
        --------------------------------------------------------------------------
        Input:
            self
        --------------------------------------------------------------------------
        Output:
            None        
        --------------------------------------------------------------------------
        Description:
            We remove officers from self.officers whose nodes are about to be arrested.
            This function serves two main purposes:
                (1) make sure we don't use officers whose lead has been removed
                    from the graph
                (2) update visualization attributes for easy debugging
        """
        #LIST OF NODES###########################################
        self.removed_nodes = officer.get_nodes_arrested()
        #########################################################

        #########################################################
        """
        Here we check if an officer is in a portion that
        is about to be removed.  We first collect officer
        instances and then remove them from the Police
        Unit attribute self.officers. 
        """
        officers_to_be_removed = [officer1 for officer1 in self.officers if officer1.get_current_node() in self.removed_nodes]
        self.officers[:]       = [officer1 for officer1 in self.officers if officer1 not in officers_to_be_removed]
        ##########################################################
        
        ##########################################################
        """
        This is for when we iterate through our strategy we do not
        use an officer that has already been removed.  Need to keep
        cumulative track of this as we do not have control what
        order the police will move in.
        """
        self.removed_officers += list(officers_to_be_removed)
        ###########################################################
        
        ##########################################################
        self.investigated = []
        ###########################################################
        
    ##################################################################################
    
    #ACTIONS##########################################################################
    def get_criminals_on_street(self):
        """
        --------------------------------------------------------------------------
        Input:
            self
            number_of_officers (int)      = Default value is the (arrival rate -1)
                                            of where the growth process is the
                                            new nodes added at each timestep for 
                                            growth.
        --------------------------------------------------------------------------
        Output:
            None
        --------------------------------------------------------------------------
        Description:
            This initializes new officers in self.officers when the officers try to
            destroy the syndicate.
        """
        number_officers_sent_out = self.officers_sent_out
        
        #MAKE NEW LEAVES < NEW OFFICERS############################################
        n = len(self.ptree.get_leaves())
        if number_officers_sent_out > n :
            number_officers_sent_out = n
        ###########################################################################
        
        #APPEND OFFICERS TO SELF.OFFICERS##########################################
        leaves_to_be_investigated = random.sample(self.ptree.get_leaves(), number_officers_sent_out) #picking without replacement
        for leaf in leaves_to_be_investigated:
            self.officers.append(Officer(self.ptree, leaf))
        ###########################################################################
        
    def go_for_root(self):
        """
        THESE ARE THE STRATEGIES IMPLEMENTED.  See the Strategies.txt for details.
        
        --------------------------------------------------------------------------
        Input:
            self
        --------------------------------------------------------------------------
        Output:
            None        
        --------------------------------------------------------------------------
        Description:
            This is the MAIN function of this class PoliceUnit.  This completes a
            timestep for the PoliceUnit allowing each officer to move and then
            arrest.  A subtle implementation detail is that the officers
            go in the order that they are chosen (which is random) and then
            go for the root.  They arrest in order and thus may miss out on 
            opportunity if they "communicated" prior to this arrest.
            
            Also, a good practice for implementing new stragies is to implement
            it in go_for_root_with_visual() and then copy it to this portion after.
            
            Strategy 1 requires a slight modifcation depending on whether a cost experiment is
            performed.
        """
        
        if self.strategy == 0:
            """
            Investigate self.number_of_investigations_for_strategy_SI times
            and then arrest.
            
            If you end at leaf, stop and the investigation stops.
            """
            #INIALIZE ATTRIBUTES###############################
            self.removed_officers =[]
            self.ptree.new_nodes = []
            ###################################################
            
            #SELECT NODES TO INVESTIGATE#######################
            self.get_criminals_on_street()
            for officer in self.get_officers():
                self.total_investigations += 1
                self.investigated += officer.get_path()
            ###################################################
            
            #EACH OFFICER MOVES################################
            for officer in self.get_officers():
                if officer in self.removed_officers:
                    pass
                else:                    
                    for k in range(self.number_of_investigations_for_strategy_SI):
                        #INVESTIGATE#########################################
                        officer.investigate()
                        self.total_investigations += 1
                        self.investigated = officer.get_path()
                        #####################################################
                
                        #####################################################
                        if officer.get_current_node() == 0:
                            self.total_arrests += 1
                            self.root_found = True
                            return
                        #####################################################
                        
                        #####################################################
                        if self.ptree.out_degree(officer.get_current_node())==0:
                            #leaves are not arrested so that resembles investigation
                            #strategy for large number of investigations
                            
                            #NO MORE INVESTIGATION###############################
                            self.investigated = []
                            #####################################################
                        
                            #REMOVE OFFICER######################################
                            self.officers.remove(officer)
                            #####################################################
                            
                            return
                        #####################################################
                    
                
                    #BEFORE ARREST#######################################
                    self.total_arrests += 1
                    self.remove_extra_officers_before_arrest(officer)
                    #####################################################
                
                
                    #AFTER ARREST########################################
                    officer.arrest()
                    #####################################################
                    
        elif self.strategy ==1:
            """
            Only investigate.  To optimize we have an officer characteristic that is officer.get_hope()
            if there is still a path to the root assuming self-avoidance.  If not, then this is False and if
            so, then True.  The cost experiment requires us to do a complete round and hence the cotrol flow
            statements related to this.  This is the only strategy where this is relevant.
            
            For this specific case, we include self.cost_experiment (bool)
            """
            self.ptree.new_nodes = []
            self.removed_officers =[]  #Not used here, but kept for continuity 
            
            #GET NEW CRIMINALS TO INVESTIGATE####################
            self.get_criminals_on_street()
            for officer in self.get_officers():
                self.total_investigations += 1
                self.investigated += officer.get_path()
            #####################################################
            
        
            for officer in self.get_officers():
                if not self.cost_experiment:
                
                    while(officer.get_hope() and officer.current_node != 0):
                
                        #INVESTIGATE#########################################
                        officer.investigate()
                        self.total_investigations +=1
                        self.investigated = officer.get_path()
                        #####################################################
            
            
                    #WHY DID THE INVESTIGATION STOP!######################
                    if officer.get_current_node() == 0:
                        self.total_arrests += 1
                        self.root_found = True
                        return
                    else:
                        self.investigated = []
                        self.officers.remove(officer)
                    #####################################################
                
                else:
                    while(officer.get_current_node() != 0 and (self.ptree.out_degree(officer.get_current_node()) != 0 or len(self.investigated) == 1)):
                            
                        #INVESTIGATE#########################################
                        officer.investigate()
                        self.total_investigations +=1
                        self.investigated = officer.get_path()
                        #####################################################
        
        
                    #WHY DID THE INVESTIGATION STOP!######################
                    if officer.get_current_node() == 0:
                        self.total_arrests += 1
                        self.root_found = True
                        return
                    else:
                        self.investigated = []
                        self.officers.remove(officer)
                    #####################################################
            
                
        
                
        elif self.strategy == 2:
            """
            Only Arrest (now deprecated by straegy 0 with self.number_of_investigations_for_strategy_SI = 0)
            """
            #INITIALIZE ATTRIBUTES################################
            self.ptree.new_nodes = []
            self.removed_officers =[]
            #####################################################

            #GET NEW CRIMINALS TO INVESTIGATE####################
            self.get_criminals_on_street()
            for officer in self.get_officers():
                self.investigated += officer.get_path()
            #####################################################
            
            
            for officer in self.get_officers():
                if officer in self.removed_officers:
                    pass
                else:  
                    #####################################################
                    if officer.get_current_node() == 0:
                        self.root_found = True
                        #print "ROOT FOUND"
                        return
                    #####################################################
                    
                    #BEFORE ARREST#######################################
                    self.remove_extra_officers_before_arrest(officer)
                    #####################################################
                    
                    
                    #AFTER ARREST########################################
                    officer.arrest()
                    #####################################################
                    
        elif self.strategy ==3:
            """
            Degree Threshold.  We arrest only when the total
            degree exceeds this threshold.  We view the police
            as slightly spiteful people, so when they complete
            a walk and end up at a leaf, they do nothing and must
            wait until the next round.
            
            Also note that the degree threshold is a total degree
            calculation including the previous edge the police
            entered from.
            """
            
            #INITIALIZE ATTRIBUTES################################
            self.ptree.new_nodes = []
            self.removed_officers =[]
            #####################################################

            #GET NEW CRIMINALS TO INVESTIGATE####################
            self.get_criminals_on_street()
            for officer in self.get_officers():
                self.total_investigations += 1
                self.investigated += officer.get_path()
            #####################################################
            
            
            for officer in self.get_officers():
                if officer in self.removed_officers:
                    pass
                else:                    
                    while( officer.get_current_node() != 0   and \
                           self.ptree.degree(officer.get_current_node()) < self.degree_threshold and \
                           officer.investigation_must_end == False 
                         ):
                         
                        #INVESTIGATE############################################
                        self.total_investigations += 1
                        officer.investigate()
                        self.investigated = officer.get_path()
                        ########################################################
                        
                    
                    #ROOT FOUND!################################################
                    if officer.get_current_node() == 0:
                        self.total_arrests += 1
                        self.root_found = True
                        return
                    #############################################################
                    
                    #REACHED LEAF################################################
                    elif self.ptree.out_degree(officer.get_current_node()) == 0 and len(self.investigated) != 1: #if q ==1, then leaves will ALWAYS 
                                                                                                                 #meet our degree threshold too so we should arrest
                                                                                                                 #the first node we see, which will be a leaf.
                        
                        #NO MORE INVESTIGATION###############################
                        self.investigated = []
                        #####################################################
                        
                        #REMOVE OFFICER######################################
                        self.officers.remove(officer)
                        #####################################################
                    
                    #HIGH DEGREE################################################
                    else:
                        """
                        This strategy removes high degree nodes! 
                        """
                        #NO MORE INVESTIGATION###############################
                        self.investigated = []
                        #####################################################
                        
                        #BEFORE ARREST#######################################
                        self.total_arrests += 1
                        self.remove_extra_officers_before_arrest(officer)
                        #####################################################
                    
                        #AFTER ARREST########################################
                        officer.arrest()
                        #####################################################
                    
                      
            
    ##################################################################################
            
    #GETTERS##########################################################################
    def get_officers(self):
        """
        Output:  returns a copy of self.officers (list of officers)
        """
        return list(self.officers)
    ##################################################################################
    
    #VISUALS##########################################################################
    def go_for_root_with_visuals(self):
        """
        See go for root!
        
        ZEN OF VISUALIZATION: visual() then timestep+=1.
        """
        if self.strategy == 0:
            """
            Walk once, then arrest
            """
            
            #INITIALIZE ATTRIBUTES################################
            self.ptree.new_nodes = []
            self.removed_officers =[]
            #####################################################

            #GET NEW CRIMINALS TO INVESTIGATE####################
            self.get_criminals_on_street()
            for officer in self.get_officers():
                self.total_investigations += 1
                self.investigated += officer.get_path()
            #####################################################
            
            #VISUAL##############################################
            print 'Get Criminals Off the street'
            self.ptree.visual(PU = self)
            self.ptree.timestep+=1
            #####################################################
            
            for officer in self.get_officers():
                if officer in self.removed_officers:
                    pass
                else:
                    for k in range(self.number_of_investigations_for_strategy_SI):                    
                        
                        #INVESTIGATE#########################################
                        officer.investigate()
                        self.total_investigations += 1
                        self.investigated = officer.get_path()
                        #####################################################
                                            
                        #VISUAL##############################################
                        print 'Investigate'
                        self.ptree.visual(PU = self)
                        self.ptree.timestep+=1
                        #####################################################
                    
                        #####################################################
                        if officer.get_current_node() == 0:
                            self.total_arrests += 1
                            self.root_found = True
                            #print "ROOT FOUND"
                            return
                        #####################################################
                        
                        #####################################################
                        if self.ptree.out_degree(officer.get_current_node())==0:
                            #leaves are not arrested
                            
                            #NO MORE INVESTIGATION###############################
                            self.investigated = []
                            #####################################################
                        
                            #REMOVE OFFICER######################################
                            self.officers.remove(officer)
                            #####################################################
                            
                            return
                            #print "ROOT FOUND"
                        #####################################################
                    
                    #BEFORE ARREST#######################################
                    self.remove_extra_officers_before_arrest(officer)
                    #####################################################
                    
                    #VISUAL##############################################
                    print 'Before arrest'
                    self.total_arrests += 1
                    self.ptree.visual(PU = self)
                    self.ptree.timestep+=1
                    #####################################################
                    
                    #AFTER ARREST########################################
                    officer.arrest()
                    #####################################################
                    
                    #VISUAL##############################################
                    print 'after arrest'
                    self.ptree.visual(PU = self)
                    self.ptree.timestep+=1
                    #####################################################
        
        elif self.strategy == 1:
            """
            Only walk
            """
            self.ptree.new_nodes = []
            self.removed_officers =[]   #Not used here, but kept for continuity
            
            #GET NEW CRIMINALS TO INVESTIGATE####################
            self.get_criminals_on_street()
            for officer in self.get_officers():
                self.total_investigations += 1
                self.investigated += officer.get_path()
            #####################################################
            
            #VISUAL##############################################
            print 'Get Criminals Off the street'
            self.ptree.visual(PU = self)
            self.ptree.timestep+=1
            #####################################################
            
            for officer in self.get_officers():                   
                if not self.cost_experiment:
                    while(officer.get_hope() and officer.current_node != 0):
                    
                        #INVESTIGATE#########################################
                        officer.investigate()
                        self.total_investigations += 1
                        self.investigated = officer.get_path()
                        #####################################################
                
                        #VISUAL##############################################
                        print 'Investigate'
                        self.ptree.visual(PU = self)
                        self.ptree.timestep+=1
                        #####################################################
                
                    #WHY DID THE INVESTIGATION STOP!######################
                    if officer.get_current_node() == 0:
                        self.total_arrests += 1
                        self.root_found = True
                        #print "ROOT FOUND"
                        return
                    else:
                        self.investigated = []
                        self.officers.remove(officer)
                    #####################################################
                
                    #VISUAL##############################################
                    print 'Investigation Failed :('
                    self.ptree.visual(PU = self)
                    self.ptree.timestep+=1
                    #####################################################
            else:
                while(officer.get_current_node() != 0 and (self.ptree.out_degree(officer.get_current_node()) != 0 or len(self.investigated) == 1)):
                    print self.ptree.out_degree(officer.get_current_node())
                    #INVESTIGATE#########################################
                    officer.investigate()
                    self.total_investigations += 1
                    self.investigated = officer.get_path()
                    #####################################################
                
                    #VISUAL##############################################
                    print 'Investigate'
                    self.ptree.visual(PU = self)
                    self.ptree.timestep+=1
                    #####################################################
                
                #WHY DID THE INVESTIGATION STOP!######################
                if officer.get_current_node() == 0:
                    self.total_arrests += 1
                    self.root_found = True
                    #print "ROOT FOUND"
                    return
                else:
                    self.investigated = []
                    self.officers.remove(officer)
                #####################################################
                
                #VISUAL##############################################
                print 'Investigation Failed :('
                self.ptree.visual(PU = self)
                self.ptree.timestep+=1
                #####################################################
        
                    
        elif self.strategy == 2:
            """
            Only Arrest (deprecated)
            """
            
            #INITIALIZE ATTRIBUTES################################
            self.ptree.new_nodes = []
            self.removed_officers =[]
            #####################################################

            #GET NEW CRIMINALS TO INVESTIGATE####################
            self.get_criminals_on_street()
            for officer in self.get_officers():
                self.investigated += officer.get_path()
            #####################################################
            
            #VISUAL##############################################
            print 'Get Criminals Off the street'
            self.ptree.visual(PU = self)
            self.ptree.timestep+=1
            #####################################################
            
            for officer in self.get_officers():
                if officer in self.removed_officers:
                    pass
                else:  
                    #####################################################
                    if officer.get_current_node() == 0:
                        self.root_found = True
                        #print "ROOT FOUND"
                        return
                    #####################################################
                    
                    #BEFORE ARREST#######################################
                    self.remove_extra_officers_before_arrest(officer)
                    #####################################################
                    
                    #VISUAL##############################################
                    print 'Before arrest'
                    self.ptree.visual(PU = self)
                    self.ptree.timestep+=1
                    #####################################################
                    
                    
                    #AFTER ARREST########################################
                    officer.arrest()
                    #####################################################
                    
                    #VISUAL##############################################
                    print 'after arrest'
                    self.ptree.visual(PU = self)
                    self.ptree.timestep+=1
                    #####################################################
    
        elif self.strategy == 3:
            """
            Degree Threshold.  We arrest only when the total
            degree exceeds this threshold.  We view the police
            as slightly spiteful people, so when they complete
            a walk and end up at a leaf, they arrest that leaf.
            
            Also note that the degree threshold is a total degree
            calculation including the previous edge the police
            entered from.
            """
            
            #INITIALIZE ATTRIBUTES################################
            self.ptree.new_nodes = []
            self.removed_officers =[]
            #####################################################

            #GET NEW CRIMINALS TO INVESTIGATE####################
            self.get_criminals_on_street()
            for officer in self.get_officers():
                self.total_investigations += 1
                self.investigated += officer.get_path()
            #####################################################
            
            #VISUAL##############################################
            print 'Get Criminals Off the street'
            self.ptree.visual(PU = self)
            self.ptree.timestep+=1
            #####################################################
            
            for officer in self.get_officers():
                if officer in self.removed_officers:
                    pass
                else:                    
                    
                    while( officer.get_current_node() != 0   and \
                           self.ptree.degree(officer.get_current_node()) < self.degree_threshold and \
                           officer.investigation_must_end == False 
                         ):
                        #INVESTIGATE#########################################
                        officer.investigate()
                        self.total_investigations += 1
                        self.investigated = officer.get_path()
                        #####################################################
                        
                        
                        #INVESTIGATION ENDS!#################################
                        if officer.investigation_must_end:
                            break
                        #####################################################
                        
                        #VISUAL##############################################
                        print 'Investigate'
                        self.ptree.visual(PU = self)
                        self.ptree.timestep+=1
                        #####################################################
                    
                    #####################################################
                    if officer.get_current_node() == 0:
                        self.total_arrests += 1
                        self.root_found = True
                        #print "ROOT FOUND"
                        return
                    #####################################################
                    elif self.ptree.out_degree(officer.get_current_node()) == 0 and len(self.investigated) != 1: #if q ==1, then leaves will ALWAYS 
                                                                                                                 #meet our degree threshold too so we should arrest
                                                                                                                 #the first node we see, which will be a leaf.
                                                
                        #NO MORE INVESTIGATION###############################
                        self.investigated = []
                        #####################################################
                        
                        #REMOVE OFFICER######################################
                        self.officers.remove(officer)
                        #####################################################
                    
                    else:
                        """
                        This strategy will either high
                        degree node.
                        """
                        #NO MORE INVESTIGATION###############################
                        self.investigated = []
                        #####################################################
                        
                        #BEFORE ARREST#######################################
                        self.total_arrests += 1
                        self.remove_extra_officers_before_arrest(officer)
                        #####################################################
                    
                        #VISUAL##############################################
                        print 'Before arrest'
                        self.ptree.visual(PU = self)
                        self.ptree.timestep+=1
                        #####################################################
                    
                    
                        #AFTER ARREST########################################
                        officer.arrest()
                        #####################################################
                    
                        #VISUAL##############################################
                        print 'after arrest'
                        self.ptree.visual(PU = self)
                        self.ptree.timestep+=1
                        #####################################################
            
    ##################################################################################

##########################################################################################################################################################################    

##########################################################################################################################################################################    
class PPTree(object):
    """
    This is a (P)oliceUnit and (P)referential (Tree) that simulates the growth of a tree
    while the Police Unit removes nodes after a growth timestep.  The two stopping conditions
    are:
        (1)  PoliceUnit reaches the root (PoliceUnit wins)
        (2)  The ptree reaches its maximum allowable nodes (ptree wins)
    """
    def __init__(self,  arrival_parameters = (20, ), max_allowable_nodes = 100, seed = (3,2), strategy = 0, officers_sent_out = 1, max_number_of_allowed_rounds = 1000, degree_threshold = None, tree_type = PTree, number_of_investigations_for_strategy_SI = 1, cost_experiment = False):
        """
        --------------------------------------------------------------------------
        Input:
            self
            seed (int, int)          (height, degree)
                                     Syndicate is initialized to be a complete tree
                                     of given height and degree.
            strategy (int)           The strategy (elaborated in the text file) that
                                     will be pursed by the PoliceUnit.
            max_allowable_nodes(int) Max allowable nodes for a growing syndicate.
            tree_type (class)        Either PTree or PruneTruee
        --------------------------------------------------------------------------
        Output:
            None
        --------------------------------------------------------------------------
        Description:
            self.PT (PTree)         This is the preferential tree
            self.PU (PoliceUnit)    This is the police unit
            
        Refer to the attributes above to see how we keep track of growth\
        and pursuit parameters.
        """
        self.PT = tree_type(arrival_parameters = arrival_parameters, \
                            max_allowable_nodes =max_allowable_nodes, \
                            seed = seed)
        self.PU = PoliceUnit(self.PT, \
                             strategy = strategy,\
                             officers_sent_out = officers_sent_out,\
                             degree_threshold = degree_threshold,\
                             number_of_investigations_for_strategy_SI = number_of_investigations_for_strategy_SI,\
                             cost_experiment = cost_experiment)
        self.max_number_of_allowed_rounds = max_number_of_allowed_rounds
        self.round_number = 1
        
    
    #MOVES############################################################################
    def growth_and_pursuit(self):
        """
        --------------------------------------------------------------------------
        Input:
            self
        --------------------------------------------------------------------------
        Output:
            None
        --------------------------------------------------------------------------
        Description:
            This grows the graph while the PoliceUnit pursues the root.
        """
        while (self.PU.root_found != True and self.PT.number_of_nodes() < self.PT.get_max_allowable_nodes() and self.round_number <= self.max_number_of_allowed_rounds):
            self.PU.go_for_root()
            if self.PU.root_found == True:
                break
            self.PT.add_nodes()
            self.round_number += 1
    ##################################################################################
            
    #VISUALS##########################################################################  
    def growth_and_pursuit_with_visuals(self):
        """
        See growth_and_pursuit()!  Same onkly with visuals()
        """
        self.PT.seed_visual()
        while (self.PU.root_found != True and self.PT.number_of_nodes() < self.PT.get_max_allowable_nodes() and self.round_number <= self.max_number_of_allowed_rounds):
            self.PU.go_for_root_with_visuals()
            if self.PU.root_found == True:
                break
            self.PT.add_nodes_with_visuals()                
            self.round_number
    ##################################################################################

##########################################################################################################################################################################    

if __name__== "__main__":  
    game = PPTree(max_allowable_nodes = 5000)
    cProfile.run('game.growth_and_pursuit()')
    

    