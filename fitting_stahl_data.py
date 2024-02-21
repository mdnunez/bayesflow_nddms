# Record of Revisions
#
# Date              Programmers                 Descriptions of Change
# ====            ================              ======================
# 14-Feb-2024     Michael D. Nunez               Original code
# 19-Feb-2024     Michael D. Nunez    Different model, same # of parameters

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

# Python references
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
# https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#copy-on-write-chained-assignment

# IMPORTS

import os
import numpy as np
import pandas as pd
import seaborn as sns
import bayesflow as bf
import matplotlib.pyplot as plt
from pyhddmjagsutils import (
    plot_posterior2d, 
    jellyfish
)
from single_trial_alpha_standard import (
    trainer,
    configurator,
    amortizer
)
# from single_trial_alpha_not_scaled import (
#     trainer,
#     configurator,
#     amortizer
# )

model_name = 'single_trial_alpha_standard'
#model_name = 'single_trial_alpha_not_scaled'


# DATA LOADING
explore = False

# Load base data from Mattes et al. (2022)
# Originally data from Stahl et al. (2015)
base_df = pd.read_csv('stahl_data/base_data.csv')

# Obtain summary measures
nsubs = np.size(np.unique(base_df['subj_idx']))


# Explore the original data
if explore:
    print(base_df.head())
    print(base_df.info())
    where_first = (base_df['subj_idx'] == 101)
    first_Pe = base_df['pre_Pe'][where_first]
    print(np.mean(first_Pe))
    print(np.std(first_Pe))
    print(np.any(np.isnan(base_df['pre_Pe'])))
    plt.figure()
    plt.hist(first_Pe)
    first_Ne = base_df['pre_Ne'][where_first]
    print(np.mean(first_Ne))
    print(np.std(first_Ne))
    print(np.any(np.isnan(base_df['pre_Ne'])))
    plt.figure()
    plt.hist(first_Ne)


# Calculate residuals to remove effect of Ne/c component from Pe/c component
# Results in the Pe/c component that has influence of the Ne/c component removed
# See paper by Mattes et al. (2022)

base_df['normalized_Ne'] = np.nan
base_df['pre_Pe_no_Ne'] = np.nan
base_df['normalized_pre_Pe_no_Ne'] = np.nan
base_df['alpha_like_Pe'] = np.nan
for part in np.unique(base_df['subj_idx']):
    these_trials = (base_df['subj_idx'] == part)
    x = base_df['pre_Ne'][these_trials]
    y = base_df['pre_Pe'][these_trials]
    coefficients = np.polyfit(x, y, deg=1) # Simple linear regression
    pred_y = np.polyval(coefficients, x)
    residuals = y - pred_y
    # base_df['pre_Pe_no_Ne'][these_trials] = residuals  # Bad, see Pandas ref
    base_df.loc[these_trials, 'pre_Pe_no_Ne'] = residuals
    normalized_Pe = (residuals - np.mean(residuals))/np.std(residuals)
    base_df.loc[these_trials, 'normalized_pre_Pe_no_Ne'] = normalized_Pe
    normalized_Ne = x/np.std(x) # Do not shift the data to get positive values
    base_df.loc[these_trials, 'normalized_Ne'] = normalized_Ne
    alpha_like_Pe = (normalized_Pe + 3)/3
    base_df.loc[these_trials, 'alpha_like_Pe'] = alpha_like_Pe



# Explore the newly created data
if explore:
    print(base_df.head())
    print(base_df.info())
    where_first = (base_df['subj_idx'] == 1)
    first_norm = base_df['alpha_like_Pe'][where_first]
    print(np.mean(first_norm))
    print(np.std(first_norm))
    print(np.any(np.isnan(base_df['alpha_like_Pe'])))
    plt.figure()
    plt.hist(first_norm)
    sub_df = base_df.iloc[:, -6:]
    print(sub_df.head())
    print(sub_df.info())
    corr_matrix = sub_df.corr()
    print(corr_matrix)
    plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)


# MODEL FITTING

# Fit the model to the data
status = trainer.load_pretrained_network()

# Calculate choice*rt for model fitting
base_df['choicert'] = base_df['rt']*(2*base_df['response'] - 1)

# Create numpy array of necessary data
# base_data_bf = np.array(base_df[['choicert','alpha_like_Pe']]) # Original to fit single_trial_alpha_not_scaled.py
base_data_bf = np.array(base_df[['choicert','normalized_pre_Pe_no_Ne']]) # New to fit single_trial_alpha_standard.py

if explore:
    these_trials = (base_df['subj_idx'] == 1)
    n_trials = np.sum(these_trials)
    sub_data = base_data_bf[these_trials,]
    obs_dict = {'sim_data': sub_data[np.newaxis,:,:], 
    'sim_non_batchable_context': n_trials, 'prior_draws': None}
    configured_dict = configurator(obs_dict)
    # Obtain posterior samples
    num_posterior_draws = 10000
    post_samples = amortizer.sample(configured_dict, num_posterior_draws)
    plt.figure()
    jellyfish(post_samples.T[:,:,None])


# Fit the model per participant and keep track of posterior distributions
num_posterior_draws = 1000
all_posteriors = np.ones((nsubs, num_posterior_draws, 8))*np.nan
part_track = 0
for part in np.unique(base_df['subj_idx']):
    these_trials = (base_df['subj_idx'] == part)
    print(f'Fitting participant {part}.')
    n_trials = np.sum(these_trials)
    sub_data = base_data_bf[these_trials,]
    obs_dict = {'sim_data': sub_data[np.newaxis,:,:], 
    'sim_non_batchable_context': n_trials, 'prior_draws': None}

    # Make sure the data matches that configurator
    configured_dict = configurator(obs_dict)

    # Obtain posterior samples
    post_samples = amortizer.sample(configured_dict, num_posterior_draws)

    all_posteriors[part_track, :, 0:7] = post_samples
    part_track += 1

# Calculate percentage of cognitive variance explained
data1_cognitive_var_samples = all_posteriors[:, :, 4]**2
data1_total_var_samples = (data1_cognitive_var_samples + 
    all_posteriors[:, :, 6]**2)
data1_cognitive_prop_samples = (data1_cognitive_var_samples / 
    data1_total_var_samples)
all_posteriors[:, :, 7] = data1_cognitive_prop_samples


# Plot the results
print('Making jellyfish plots.')
plot_path = f"data_plots/{model_name}"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

plt.figure()
jellyfish(all_posteriors[:,:,0,None])
plt.xlabel('Drift rate (evidence units / sec)')
plt.ylabel('Participant')
plt.savefig(f'{plot_path}/{model_name}_Drift_stahl_base.png')
plt.close()


plt.figure()
jellyfish(all_posteriors[:,:,1,None])
plt.xlabel('Boundary (evidence units)')
plt.ylabel('Participant')
plt.savefig(f'{plot_path}/{model_name}_Boundary_stahl_base.png')
plt.close()

plt.figure()
jellyfish(all_posteriors[:,:,2,None])
plt.xlabel('Relative Start Point (evidence units)')
plt.ylabel('Participant')
plt.savefig(f'{plot_path}/{model_name}_StartPoint_stahl_base.png')
plt.close()

plt.figure()
jellyfish(all_posteriors[:,:,3,None])
plt.xlabel('Non-decision time (sec)')
plt.ylabel('Participant')
plt.savefig(f'{plot_path}/{model_name}_NDT_stahl_base.png')
plt.close()

plt.figure()
jellyfish(all_posteriors[:,:,4,None])
plt.xlabel('Boundary across-trial std (evidence units)')
plt.ylabel('Participant')
plt.savefig(f'{plot_path}/{model_name}_BoundarySDT_stahl_base.png')
plt.close()

plt.figure()
jellyfish(all_posteriors[:,:,5,None])
plt.xlabel('Diffusion coefficient (evidence units)')
plt.ylabel('Participant')
plt.savefig(f'{plot_path}/{model_name}_DC_stahl_base.png')
plt.close()

plt.figure()
jellyfish(all_posteriors[:,:,6,None])
plt.xlabel('Noise in Pe not related to boundary')
plt.ylabel('Participant')
plt.savefig(f'{plot_path}/{model_name}_PeNoise_stahl_base.png')
plt.close()

plt.figure()
jellyfish(all_posteriors[:,:,7,None])
plt.xlabel('Proportion of Pe related to single-trial boundary')
plt.ylabel('Participant')
plt.savefig(f'{plot_path}/{model_name}_PeProportion_stahl_base.png')
plt.close()

print('Making 2D plots.')
nplots = 18
scatter_color = '#ABB0B8'
plot_posterior2d(all_posteriors[0:nplots, :, 5].squeeze(),
    all_posteriors[0:nplots, :, 1].squeeze(),
   ['Diffusion coefficient', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_dc_stahl_base.png")

plot_posterior2d(all_posteriors[0:nplots, :, 0].squeeze(),
    all_posteriors[0:nplots, :, 1].squeeze(),
   ['Drift rate', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_drift_stahl_base.png")

plot_posterior2d(all_posteriors[0:nplots, :, 5].squeeze(),
    all_posteriors[0:nplots, :, 0].squeeze(),
   ['Diffusion coefficient', 'Drift rate'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_drift_dc_stahl_base.png")


# Plot a 3D joint posterior
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

main_color = '#332288'
secondary_color = '#ABB0B8'

# By default plot the 3D posterior for only one participant
part_indx = 0 #Participant 1
this_part = np.unique(base_df['subj_idx'])[part_indx]
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

plt.savefig(f"{plot_path}/{model_name}_3d_posterior_stahl_base.png", 
    dpi=300, bbox_inches="tight", pad_inches=0.5)
