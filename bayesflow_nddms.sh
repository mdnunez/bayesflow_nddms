#!/bin/bash
#Set job requirements
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -n 1
#SBATCH -t 30:00:00
 

#To run on Snellius (and other servers with SLURM) use:
#dos2unix bayesflow_nddms.sh && sbatch bayesflow_nddms.sh

# Model to run
model="single_trial_alpha_mean"

# Echo for first slurm output check
echo -e "Running script $model.py"

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0 # Does this help?

# Now follow these steps
# https://github.com/stefanradev93/BayesFlow/blob/master/INSTALL.rst

#Copy local BayesFlow to avoid using different package versions
cp -r $HOME/BayesFlow "$TMPDIR"

# Install BayesFlow and dependencies
pip install --user "$TMPDIR"/BayesFlow

# Install numba
pip install --user numba
 
#Copy input files to scratch
cp -r $HOME/bayesflow_nddms "$TMPDIR"

#Change directory to bayesflow_nddms
cd "$TMPDIR"/bayesflow_nddms

#List contents of directory for debugging
ls

#Execute the Python program
python $model.py

#Make the directories if they do not exist with -p flag
mkdir -p $HOME/bayesflow_nddms/recovery_plots/$model
mkdir -p $HOME/bayesflow_nddms/checkpoint/$model
 
#Copy output directories from scratch to home
cp -r recovery_plots/$model/* $HOME/bayesflow_nddms/recovery_plots/$model/
cp -r checkpoint/$model/* $HOME/bayesflow_nddms/checkpoint/$model/