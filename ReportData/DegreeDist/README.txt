Degree Distribution as Effected By Maximum Allowable Nodes Increases:

Experiments: 100 
Max Allowable Nodes Interval:  500 1500
Preference: 
    1/(distance_to_closest_leaf+1)
    
Arrival parameters: (3,)

Regarding Degree_Distribution.pkl file:
The key 'degree_dict' is the dictionary with key: degree and value: number of occurancesin all experiments! 
The key 'exper_dict' is the dictionary for the number of experiments each degree showed up
The degree frequency is obtained by: [float(degree_dict[d])/exper_dict[d] for d in degrees]