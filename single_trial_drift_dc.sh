#!/bin/bash
#Set job requirements
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -n 1
#SBATCH -t 8:00:00
 

#To run on Senllius use:
#dos2unix single_trial_drift_dc.sh && sbatch single_trial_drift_dc.sh

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Now follow these steps
# https://github.com/stefanradev93/BayesFlow/blob/master/INSTALL.rst

# Install BayesFlow and dependencies form github
pip install --user git+https://github.com/stefanradev93/bayesflow

# Install numba
pip install --user numba
 
#Copy input files to scratch
cp -r $HOME/bayesflow_nddms "$TMPDIR"

#Change directory to bayesflow_nddms
cd "$TMPDIR"/bayesflow_nddms

#List contents of directory for debugging
ls

#Execute the Python program
python single_trial_drift_dc.py
 
#Copy output directories from scratch to home
cp -r recovery_plots/single_trial_drift_dc/* $HOME/bayesflow_nddms/recovery_plots/single_trial_drift_dc/
cp -r checkpoint/single_trial_drift_dc/* $HOME/bayesflow_nddms/checkpoint/single_trial_drift_dc/