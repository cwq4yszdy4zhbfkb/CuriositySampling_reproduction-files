# Experiment Reproduction Files

Welcome to our repository of files necessary to reproduce experiments from the Results section. Please note that the trajectories for RfaH-CTD, WLALL5, and Chigolin are not included due to their large size (hundreds of GBs).

Each directory contains the following files:
    * `chignolin_curiosity` - Files required to generate the Curiosity Sampling trajectory presented in section 2.3 of the main manuscript.
    * `chignolin_hyperparameter_and_MSM_related_data` - Python script for hyperparameter search for Chignolin MSM, along with a Jupyter notebook to train MSM and show all the corresponding properties of it and Chignolin MD simulation (RMSD, Q coordinate).
    * `2D_potential_curiosity` - Scripts to generate 2D potential simulations, including code for the 2D potential. Additionally, $C_{t_{max}}$ configurations and trajectories for each random seed (1 to 10) of the 2D potential simulation.
    * `RfaH-CTD_curiosity` - Scripts and starting structures to generate Curiosity Sampling simulations for RfaH-CTD, presented in section 2.2 of the manuscript.
    * `WLALL5_curiosity` - Scripts and starting structures to generate Curiosity Sampling simulations for WLALL5 protein, presented in section 2.2 of the manuscript.
    * `RfaH-CTD_ground_truth` - Scripts and starting structures to generate Ground Truth data for RfaH-CTD, presented in section 2.2 of the manuscript.
    * `WLALL5_ground_truth` - Scripts and starting structures to generate Ground Truth data for WLALL5, presented in section 2.2 of the manuscript.
    * `RfaH-CTD_adaptive_sampling` - Scripts and starting structures to generate Adaptive Sampling simulations for RfaH-CTD, presented in section 2.2 of the manuscript.
    * `RfaH-CTD_adaptive_bandit` - Scripts and starting structures to generate Adaptive Bandit simulations for RfaH-CTD, presented in section 2.2 of the manuscript.
    * `WLALL5_adaptive_sampling` - Scripts and starting structures to generate Adaptive Sampling simulations for WLALL5, presented in section 2.2 of the manuscript.
    * `WLALL5_adaptive_bandit` - Scripts and starting structures to generate Adaptive Bandit simulations for WLALL5, presented in section 2.2 of the manuscript.
    * `FoNM_calculations` - Scripts regarding Fraction of New Microstates (FoNM) calculations, including clustering.

Each directory contains a README.md file explaining the files inside. Most of the scripts include `run_openmm_20_1.sh` and `curiositcalc_rep20.py`. The first is a SLURM script that runs the `curiositcalc_rep20.py` Python script, which then loads and initializes Curiosity Sampling.

# Package Versions Used

Before running any of the simulations, you must install Curiosity Sampling (see the GitHub page). The following package versions were used when performing experiments from the Results section:

* python                    3.9.15
* tensorflow                2.8.2
* tensorboard               2.8.0
* tensorflow-addons         0.16.1
* tensorflow-probability    0.16.0
* pyemma                    2.5.11
* scikit-learn              1.0.2
* scipy                     1.8.0
* ray                       1.11.0
* mdtraj                    1.9.7
* parmed                    3.4.3
* cudatoolkit               11.2.2
* cudnn                     8.1.0.77
* HTMD                      2.0.5

# Curiosity Sampling installation

The repository with the algorithm can be found [here](https://github.com/cwq4yszdy4zhbfkb/CuriositySampling_review).

# Licence
This work is licensed under the CC BY-NC-SA 4.0 (see LICENSE.txt for more)

