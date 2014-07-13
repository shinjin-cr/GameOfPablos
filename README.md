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
2. Criminal (node) Density
3. Street Criminal (leaf) Distance
4. Street Criminal Growth

PursuitExperiments.py
================================================================================================================================
This is the driver that collects stats on the Dynamic Game, which is a combination of Pursuit and Growth.

Report/
================================================================================================================================
.tex files that includes a detailed explanation of the model.

ReportData/
================================================================================================================================
Data with plots that appear in Report.  Most of the files were obtained via school machines and plots were then done locally.

MuriMeetingPoster/
================================================================================================================================
This is a poster for the project.

PreliminaryBeamer/
================================================================================================================================
Slides done in the nacency of the project.  More graph theoretic language.
