# Record of Revisions
#
# Date              Programmers                 Descriptions of Change
# ====            ================              ======================
# 21-Feb-2024     Michael D. Nunez   Converted from singel_trial_alpha_standnorm

# Academic references:
#
# Mattes, A., Porth, E., & Stahl, J. (2022). Linking neurophysiological 
# processes of action monitoring to post-response speed-accuracy adjustments in 
# a neuro-cognitive diffusion model. NeuroImage, 247, 118798.
#
# Stahl, J., Acharki, M., Kresimon, M., Völler, F., & Gibbons, H. (2015). 
# Perfect error processing: Perfectionism-related variations in action 
# monitoring and error processing mechanisms. International Journal of 
# Psychophysiology, 97(2), 153-162.

# IMPORTS

import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import truncnorm
import bayesflow as bf
import matplotlib.pyplot as plt
from pyhddmjagsutils import (
    plot_posterior2d, 
    recovery,
    recovery_scatter
)
from single_trial_alpha_standnorm import (
    trainer,
    configurator,
    amortizer
)


model_name = 'single_trial_alpha_standnorm'


# DATA LOADING
explore = True

# Load base data from Mattes et al. (2022)
# Originally data from Stahl et al. (2015)
base_df = pd.read_csv('stahl_data/base_data.csv')

# Obtain summary measures
nsubs = np.size(np.unique(base_df['subj_idx']))
all_Pe = base_df['pre_Pe']
ntrials_total = np.size(all_Pe)

# Explore the original data
if explore:
    print(base_df.head())
    print(base_df.info())
    print(f'The minimum Pe is {np.min(all_Pe)}')
    print(f'The maximum Pe is {np.max(all_Pe)}')
    print(f'The mean of Pe is {np.mean(all_Pe)}')
    print(f'The standard deviation of Pe is {np.std(all_Pe)}')
    plt.figure()
    plt.hist(all_Pe)
    plt.title('Histogram of all Pe/c across participants and trials')

all_standard_Pe = (all_Pe - np.mean(all_Pe)) / np.std(all_Pe) # Input to model fits
single_trial_alphas = (all_standard_Pe + 3)/3

if explore:
    plt.figure()
    sns.kdeplot(all_standard_Pe)
    plt.title('Estimated density of all standardized Pe/c across trials')
    plt.figure()
    sns.kdeplot(single_trial_alphas)
    plt.title('Estimated density of imputed single-trial boundaries')
    num_bad_alphas = np.sum(single_trial_alphas < 0) / np.size(all_Pe)
    print(f'The percent of imputed boundaries below zero is {num_bad_alphas*100}%')
    print(f'The minimum of imputed boundaries is {np.min(single_trial_alphas)}')
    print(f'The maximum of imputed boundaries is {np.max(single_trial_alphas)}')
    print(f'The mean of imputed boundaries is {np.mean(single_trial_alphas)}')
    print(f'The std of imputed boundaries is {np.std(single_trial_alphas)}')

# Set alphas below 0 to 0
single_trial_alphas[single_trial_alphas < 0] = 0

if explore:
    print('Fixed imputed boundaries below 0 to 0')
    num_bad_alphas = np.sum(single_trial_alphas < 0) / np.size(all_Pe)
    print(f'The percent of imputed boundaries below zero is {num_bad_alphas*100}%')
    print(f'The minimum of imputed boundaries is {np.min(single_trial_alphas)}')
    print(f'The maximum of imputed boundaries is {np.max(single_trial_alphas)}')
    print(f'The mean of imputed boundaries is {np.mean(single_trial_alphas)}')
    print(f'The std of imputed boundaries is {np.std(single_trial_alphas)}')
    corr_to_data = np.corrcoef(all_Pe,single_trial_alphas)[0,1]
    print(f'The correlation of data and true imputed boundaries is {corr_to_data}')

# Impute choice-RTs based on changing trial-to-trial EEG measures

def diffusion_trial(drift, bound_trial, beta, ter, dc, dt=.01, max_steps=400.):
    """Simulates a trial from the diffusion model."""

    # trial-to-trial boundary
    if bound_trial<0:
        raise ValueError("Trial-level boundary cannot be less than zero")

    n_steps = 0.
    evidence = bound_trial * beta
 
    # Simulate a single DM path
    while ((evidence > 0) and (evidence < bound_trial) and (n_steps < max_steps)):

        # DDM equation
        evidence += drift*dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt


    if evidence >= bound_trial:
        choicert =  ter + rt  
    elif evidence <= 0:
        choicert = -ter - rt
    else:
        choicert = 0  # This indicates a missing response
    return choicert

def truncnorm_better(mean=0, sd=1, low=-10, upp=10, size=1):
    return truncnorm.rvs(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd, size=size)


# Generate simulated and imputed variables
RNG = np.random.default_rng(2024)
imputed_df = pd.DataFrame(index=range(nsubs), 
    columns=['PartID','Drift', 'Mu_Alpha', 'Beta', 'Ter', 'Var_Alpha', 'Dc'])
imputed_df.loc[:, :] = np.nan
part_track = 0
for part in np.unique(base_df['subj_idx']):
    print(f'Imputing participant {part} single-trial boundaries from Pe/c.')
    these_trials = (base_df['subj_idx'] == part)
    imputed_df.loc[part_track, 'PartID'] = part
    drift = RNG.normal(3.0, 1.0) # Set all drifts to positive
    imputed_df.loc[part_track,'Drift'] = drift 
    imputed_df.loc[part_track,'Mu_Alpha'] = np.mean(single_trial_alphas[these_trials])
    beta = RNG.beta(25.0, 25.0) # Set start points to around .5
    imputed_df.loc[part_track,'Beta'] = beta
    ter = truncnorm_better(mean=0.4, sd=0.1, low=0.0, upp=1.5)[0]
    imputed_df.loc[part_track,'Ter'] = ter
    imputed_df.loc[part_track,'Var_Alpha'] = np.var(single_trial_alphas[these_trials])
    dc = truncnorm_better(mean=1.0, sd=0.25, low=0.0, upp=10)[0] #Set DC around 1
    imputed_df.loc[part_track,'Dc'] = dc
    part_track += 1


if explore:
    print(imputed_df.head())
    print(imputed_df.info())
    plt.figure()
    sns.kdeplot(imputed_df['Drift'])
    plt.title('Simulated drift rates')
    plt.figure()
    sns.kdeplot(imputed_df['Mu_Alpha'])
    plt.title('Imputed mean boundaries')
    plt.figure()
    sns.kdeplot(imputed_df['Beta'])
    plt.title('Simulated start points')
    plt.figure()
    sns.kdeplot(imputed_df['Ter'])
    plt.title('Simulated non-decision times')
    plt.figure()
    sns.kdeplot(imputed_df['Var_Alpha'])
    plt.title('Imputed trial-to-trial variance of boundaries')
    plt.figure()
    sns.kdeplot(imputed_df['Dc'])
    plt.title('Simulated start points')


# Generate imputed choice-RTs
print(f'Imputing participant choice-RTs from Pe/c.')
imputed_choicert = np.ones((ntrials_total))*np.nan
part_track = 0
for trial in range(ntrials_total):
    this_part = base_df.loc[trial, 'subj_idx']
    part_indx = np.where(imputed_df['PartID'] == this_part)[0][0]
    choicert_draw = diffusion_trial(imputed_df.loc[part_indx,'Drift'],
        single_trial_alphas[trial], imputed_df.loc[part_indx,'Beta'],
        imputed_df.loc[part_indx,'Ter'], imputed_df.loc[part_indx,'Dc'])
    imputed_choicert[trial] = choicert_draw

if explore:
    plt.figure()
    sns.kdeplot(imputed_choicert)
    plt.title('All choice RTs')
    num_missing = np.sum(imputed_choicert == 0)
    print(f'There are {num_missing} trials generated')

########### UNFINISHED FROM HERE ######

# MODEL FITTING

# Fit the model to the data
status = trainer.load_pretrained_network()

# Create numpy array of necessary data
input_data = np.column_stack((imputed_choicert,all_standard_Pe)) 


# Fit the model per participant and keep track of posterior distributions
num_posterior_draws = 1000
all_posteriors = np.ones((nsubs, num_posterior_draws, 6))*np.nan
part_track = 0
for part in np.unique(base_df['subj_idx']):
    these_trials = (base_df['subj_idx'] == part)
    print(f'Fitting participant {part}.')
    n_trials = np.sum(these_trials)
    sub_data = input_data[these_trials,]
    obs_dict = {'sim_data': sub_data[np.newaxis,:,:], 
    'sim_non_batchable_context': n_trials, 'prior_draws': None}

    # Make sure the data matches that configurator
    configured_dict = configurator(obs_dict)

    # Obtain posterior samples
    post_samples = amortizer.sample(configured_dict, num_posterior_draws)

    all_posteriors[part_track, :, 0:6] = post_samples
    part_track += 1



# Plot the results

plot_path = f"recovery_plots/{model_name}"

param_means = all_posteriors.mean(axis=1)
# Plot true versus estimated for a subset of parameters
recovery_scatter(np.array(imputed_df[['Drift', 'Dc', 'Mu_Alpha', 'Beta', 'Ter']]),
                  param_means[:, np.array([0, 5, 1, 2, 3])],
                  ['Drift Rate', 'Diffusion Coefficient', 'Boundary',
                  'Start Point', 'Non-Decision Time'],
                  font_size=16, color='#3182bdff', alpha=0.75, grantB1=False)
plt.savefig(f"{plot_path}/{model_name}_recovery_short.png")


plt.figure()
# Use None to add singleton dimension for recovery which expects multiple chains
recovery(all_posteriors[:, :, 0, None],
    imputed_df['Drift'])
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Drift')
plt.savefig(f'{plot_path}/{model_name}_Drift_imputed.png')
plt.close()

plt.figure()
recovery(all_posteriors[:, :, 1, None],
    imputed_df['Mu_Alpha'])
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Boundary')
plt.savefig(f'{plot_path}/{model_name}_Boundary_imputed.png')
plt.close()

plt.figure()
recovery(all_posteriors[:, :, 2, None],
    imputed_df['Beta'])
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Relative Start Point')
plt.savefig(f'{plot_path}/{model_name}_StartPoint_imputed.png')
plt.close()

plt.figure()
recovery(all_posteriors[:, :, 3, None],
    imputed_df['Ter'])
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Non-decision time')
plt.savefig(f'{plot_path}/{model_name}_NDT_imputed.png')
plt.close()

plt.figure()
recovery(all_posteriors[:, :, 4, None],
    imputed_df['Var_Alpha'])
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Trial-to-trial variance in boundary')
plt.savefig(f'{plot_path}/{model_name}_boundary_var_imputed.png')
plt.close()

plt.figure()
recovery(all_posteriors[:, :, 5, None],
    imputed_df['Dc'])
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Diffusion coefficient')
plt.savefig(f'{plot_path}/{model_name}_DC_imputed.png')
plt.close()

print('Making 2D plots.')
nplots = 18
scatter_color = '#ABB0B8'
plot_posterior2d(all_posteriors[0:nplots, :, 5].squeeze(),
    all_posteriors[0:nplots, :, 1].squeeze(),
   ['Diffusion coefficient', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_dc_imputed.png")

plot_posterior2d(all_posteriors[0:nplots, :, 0].squeeze(),
    all_posteriors[0:nplots, :, 1].squeeze(),
   ['Drift rate', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_drift_imputed.png")

plot_posterior2d(all_posteriors[0:nplots, :, 5].squeeze(),
    all_posteriors[0:nplots, :, 0].squeeze(),
   ['Diffusion coefficient', 'Drift rate'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_drift_dc_imputed.png")


# Plot a 3D joint posterior
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

main_color = '#332288'
secondary_color = '#ABB0B8'

# By default plot the 3D posterior for only one participant
this_part = 101 #Participant 101
part_indx = np.where(imputed_df['PartID'] == this_part)[0][0]
print(f'Making 3D plot for Participant {this_part}.')

# Main 3D scatter plot
ax.scatter(all_posteriors[part_indx, :, 0].squeeze(),
           all_posteriors[part_indx, :, 1].squeeze(),
           all_posteriors[part_indx, :, 5].squeeze(), alpha=0.25, 
           color=main_color)

# 2D scatter plot for drift rate and boundary (xy plane) at min diffusion coefficient
min_dc = all_posteriors[part_indx, :, 5].min()
ax.scatter(all_posteriors[part_indx, :, 0].squeeze(), 
    all_posteriors[part_indx, :, 1].squeeze(), 
    min_dc, alpha=0.25, color=secondary_color)

# 2D scatter plot for drift rate and diffusion coefficient (xz plane) at max boundary
max_boundary = all_posteriors[part_indx, :, 1].max()
ax.scatter(all_posteriors[part_indx, :, 0].squeeze(), max_boundary, 
    all_posteriors[part_indx, :, 5].squeeze(), alpha=0.25, 
    color=secondary_color)

# 2D scatter plot for boundary and diffusion coefficient (yz plane) at min drift rate
min_drift_rate = all_posteriors[part_indx, :, 0].min()
ax.scatter(min_drift_rate, all_posteriors[part_indx, :, 1].squeeze(), 
    all_posteriors[part_indx, :, 5].squeeze(), alpha=0.25, 
    color=secondary_color)

ax.set_xlabel(r'Drift rate ($\delta$)', fontsize=16, labelpad=10)
ax.set_ylabel(r'Boundary ($\alpha$)', fontsize=16, labelpad=10)
ax.set_zlabel(r'Diffusion coefficient ($\varsigma$)', fontsize=16, 
    labelpad=10)

# Rotate the plot slightly clockwise around the z-axis
elevation = 20  # Default elevation
azimuth = -30   # Rotate 30 degrees counterclockwise from the default azimuth (which is -90)
ax.view_init(elev=elevation, azim=azimuth)

plt.savefig(f"{plot_path}/{model_name}_3d_posterior_imputed.png", 
    dpi=300, bbox_inches="tight", pad_inches=0.5)
