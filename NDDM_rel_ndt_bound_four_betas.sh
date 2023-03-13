#!/bin/bash
#Set job requirements
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -n 1
#SBATCH -t 8:00:00
 

#To run on Senllius use:
#dos2unix NDDM_rel_ndt_bound_four_betas.sh && sbatch NDDM_rel_ndt_bound_four_betas.sh

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

#Make the directories if they do not exist with -p flag
mkdir -p recovery_plots/NDDM_rel_ndt_bound_four_betas
mkdir -p checkpoint/NDDM_rel_ndt_bound_four_betas
 
#Copy output directories from scratch to home
cp -r recovery_plots/NDDM_rel_ndt_bound_four_betas/* $HOME/bayesflow_nddms/recovery_plots/NDDM_rel_ndt_bound_four_betas/
cp -r checkpoint/NDDM_rel_ndt_bound_four_betas/* $HOME/bayesflow_nddms/checkpoint/NDDM_rel_ndt_bound_four_betas/