PTree.py
================================================================================================================================

This file includes Four classes:

+ **PTree**: This is the Preferential Attachment Tree, which the growth model is based.  
+ **Officer**: This is the officer class.
+ **PoliceUnit**: A collection of officers--our numerical experiments only require one officer.  This is where strategies are encoded.
+ **PPTree**: Dynamic Game Object with all of the classes above.

All the files includes a visualization alternatives for debugging. 

GrowthExperiments.py
================================================================================================================================
This is the driver that collects stats on the growth properties:

1. Degree Distribution
2. Criminal (node) Density and Leaf Density
3. Street Criminal (leaf) Distance
4. Street Criminal Growth
5. Total weight

Many of these experiments are then studied as the growth parameters are varied, e.g. initial network configuration and arrival rate

PursuitExperiments.py
================================================================================================================================
This is the driver that collects stats on the Dynamic Game, which is a combination of Pursuit and Growth.  The properties are:

1. Beat #:= the max arrival rate at which the probability of Police victory = 1.
2. Cost  := the number of investigations and arrests required for police to win/to loose/on average.

We investigate these properties with respect to the different strategies the Police have.  Again, we vary the parameters to refine
our understanding of the properties above and the individual strategies.

GrowthForCluster1.py/GrowthForCluster2.py/PursuitForCluster1.py/PursuitForCluster2.py
================================================================================================================================
These are the jobs run on the school cluster.  Mainly here for record.
