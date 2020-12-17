## Code of our manuscript "An Unsupervised Bayesian Neural Network for Truth Discovery in Social Networks".##

###DataSets ###
**IMDB**
IMDB agent observations :  DataACT_Real_Modify_Level2.csv
IMDB network : YObs_Real_Filter_Edges_Modify.csv
**CF**
CF agent observations :  CF.csv
CF network : CF_topology.csv
**SP**
SP agent observations :  SP.csv
SP network : SP_topology.csv

###Codes###

Data preprocessing used to generate adjacency matrices: DataProcess_TruthDiscovery.py

Deep Truth Discovery codes for three datasets: 

**CF**: python DTD_CF.py

**IMDB**: python DTD_level2.py

**SP**: python DTD_SP.py