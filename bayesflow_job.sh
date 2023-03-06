#!/bin/bash
#Set job requirements
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 48:00:00
 

#To run on Senllius use:
#dos2unix bayesflow_job.sh && sbatch bayesflow_job.sh

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

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
cp -r recovery_plots/* $HOME/bayesflow_nddms/recovery_plots/
cp -r checkpoint/single_trial_drift_dc/* $HOME/bayesflow_nddms/checkpoint/single_trial_drift_dc/