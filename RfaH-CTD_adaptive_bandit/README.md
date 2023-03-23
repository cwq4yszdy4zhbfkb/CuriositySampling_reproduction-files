# RfaH-CTD simulations

Here is a directory from which you can repeat RfaH-CTD simulations. The directory `SEED_1` provides files and parameters for a simulation with a random seed set to 1. To repeat all 5 runs, you have to copy this directory into 4 different directories. Then inside the code (`Run_AdaptiveBandit.ipynb`), change the line `ALL_RND_SEED=1` to a number from `2 .. 5`. Then run Jupyter Notebook server on your node and start the simulation by executing Jupyter Notebook's code. 
