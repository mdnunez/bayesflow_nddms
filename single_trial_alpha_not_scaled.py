# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 16-May-23     Michael Nunez      Conversion from single_trial_alpha.py
#                          Assume noisy absolute evidence scale is observed
# 14-June-23    Michael Nunez                   Add publication text

# References:
# https://github.com/stefanradev93/BayesFlow/blob/master/docs/source/tutorial_notebooks/LCA_Model_Posterior_Estimation.ipynb
# https://github.com/stefanradev93/BayesFlow/blob/master/docs/source/tutorial_notebooks/Covid19_Initial_Posterior_Estimation.ipynb
# https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy

# Notes:
# 1) conda activate bf
# 2) Do not create checkpoint folder manually, let BayesFlow do it otherwise get a no memory.pkl error

import os
import numpy as np
from scipy.stats import truncnorm
from numba import njit
import seaborn as sns
import bayesflow as bf
import matplotlib.pyplot as plt
from pyhddmjagsutils import recovery, recovery_scatter, plot_posterior2d, jellyfish

num_epochs = 500
view_simulation = False
train_fitter = False


# Get the filename of the currently running script
filename = os.path.basename(__file__)

# Remove the .py extension from the filename
model_name = os.path.splitext(filename)[0]

if train_fitter:
    print(f'Training fitting network for model {model_name} with {num_epochs} training epochs.')


# Generative Model Specifications User Defined Functions, non-batchable

def prior_N(n_min=60, n_max=300):
    """A prior for the random number of observation"""
    return np.random.randint(n_min, n_max+1)


def truncnorm_better(mean=0, sd=1, low=-10, upp=10, size=1):
    return truncnorm.rvs(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd, size=size)


RNG = np.random.default_rng(2023)
np.set_printoptions(suppress=True)
def draw_prior():

    # drift ~ N(0, 2.0), drift rate, index 0
    drift = RNG.normal(0.0, 2.0)

    # mu_alpha ~ N(1.0, 0.5) in [0, 10], mean boundary, index 1
    mu_alpha = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=10)[0]

    # beta ~ Beta(2.0, 2.0), relative start point, index 2
    beta = RNG.beta(2.0, 2.0)

    # ter ~ N(0.5, 0.25) in [0, 1.5], non-decision time, index 3
    ter = truncnorm_better(mean=0.5, sd=0.25, low=0.0, upp=1.5)[0]

    # var_alpha ~ N(1.0, 0.5) in [0, 3], trial-to-trial variability in boundary, index 4
    var_alpha = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=3)[0]

    #dc ~ N(1.0, 0.5) in [0, 10], diffusion coefficient, index 5
    dc = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=10)[0]

    # sigma1 ~ U(0.0, 5.0),measurement noise of extdata1, index 6
    sigma1 = RNG.uniform(0.0, 5.0)

    p_samples = np.hstack((drift, mu_alpha, beta, ter, var_alpha, dc, sigma1))
    return p_samples


num_params = draw_prior().shape[0]

@njit
def diffusion_trial(drift, mu_alpha, beta, ter, var_alpha, dc, sigma1, 
    dt=.01, max_steps=400.):
    """Simulates a trial from the diffusion model."""

    # trial-to-trial boundary
    while True:
        bound_trial = mu_alpha + var_alpha * np.random.normal()
        if bound_trial>0:
            break

    n_steps = 0.
    evidence = bound_trial * beta
 
    # Simulate a single DM path
    while ((evidence > 0) and (evidence < bound_trial) and (n_steps < max_steps)):

        # DDM equation
        evidence += drift*dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt

 
    # Observe absolute measures with noise
    extdata1 = np.random.normal(1*bound_trial, sigma1)
    #extdata1 = np.random.normal(b * bound_trial, 1)

    if evidence >= bound_trial:
        choicert =  ter + rt  
    elif evidence <= 0:
        choicert = -ter - rt
    else:
        choicert = 0  # This indicates a missing response
    return choicert, extdata1

@njit
def simulate_trials(params, n_trials):
    """Simulates a diffusion process for trials ."""

    drift, mu_alpha, beta, ter, var_alpha, dc, sigma1 = params
    choicert = np.empty(n_trials)
    z1 = np.empty(n_trials)
    for i in range(n_trials):
        choicert[i], z1[i] = diffusion_trial(drift, mu_alpha, beta, ter, var_alpha, dc, sigma1)
   
    sim_data = np.stack((choicert, z1), axis=-1)
    return sim_data



# Connect via BayesFlow Wrappers
prior = bf.simulation.Prior(prior_fun=draw_prior)
experimental_context = bf.simulation.ContextGenerator(non_batchable_context_fun=prior_N)
simulator = bf.simulation.Simulator(simulator_fun=simulate_trials, 
    context_generator=experimental_context)
generative_model = bf.simulation.GenerativeModel(prior, simulator)


# Create Configurator
# We need this, since the variable N cannot be processed directly by the nets.
def configurator(sim_dict):
    """Configures the outputs of a generative model for interaction with 
    BayesFlow modules."""
    
    out = dict()
    # These will be passed through the summary network. In this case,
    # it's just the data, but it can be other stuff as well.
    data = sim_dict['sim_data'].astype(np.float32)
    out['summary_conditions'] = data
    
    # These will be concatenated to the outputs of the summary network
    # Convert N to log N since neural nets cant deal well with large numbers
    N = np.log(sim_dict['sim_non_batchable_context'])
    # Repeat N for each sim (since shared across batch), notice the
    # extra dimension needed
    N_vec = N * np.ones((data.shape[0], 1), dtype=np.float32)
    out['direct_conditions'] = N_vec
    
    # Finally, extract parameters. Any transformations (e.g., standardization)
    # should happen here.
    out['parameters'] = sim_dict['prior_draws'].astype(np.float32)
    return out


if view_simulation:
    # Plot the posterior distributions of choice RTs and EEG data
    num_test = 5000


    # Need to test for different Ns, which is what the following code does
    extdata1_means =np.empty((num_test))
    extdata1_vars = np.empty((num_test))
    rt_means = np.empty((num_test))
    choice_means = np.empty((num_test))
    np.random.seed(2023) # Set the random seed to generate the same plots every time
    for i in range(num_test):
        raw_sims = generative_model(1)
        these_sims = raw_sims['sim_data']
        extdata1_means[i] = np.mean(np.squeeze(these_sims[0, :, 1]))
        extdata1_vars[i] = np.var(np.squeeze(these_sims[0, :, 1]))
        rt_means[i] = np.mean(np.abs(np.squeeze(these_sims[0,:, 0])))
        choice_means[i] = np.mean(.5 + .5*np.sign(np.squeeze(these_sims[0,:, 0]))) #convert [1, -1] to [1, 0]


    plt.figure()
    sns.kdeplot(extdata1_means)

    plt.figure()
    sns.kdeplot(extdata1_vars)

    plt.figure()
    sns.kdeplot(rt_means)

    plt.figure()
    sns.kdeplot(choice_means)

    plt.figure()
    sns.kdeplot(np.squeeze(these_sims[0, :, 1]))

    sim_rts = np.abs(np.squeeze(these_sims[0, :, 0]))
    sim_choices = np.sign(np.squeeze(these_sims[0,:, 0]))
    # This should look like a shifted Wald
    plt.figure()
    sns.kdeplot(sim_rts[sim_choices == 1])

    # This should look like a shifted Wald
    plt.figure()
    sns.kdeplot(sim_rts[sim_choices == -1])

    plt.show(block=False)

    # Minimum RT
    print('The minimum RT is %.3f when the NDT was %.3f' % 
        (np.min(sim_rts), raw_sims['prior_draws'][0,3]))




# BayesFlow Setup
summary_net = bf.networks.InvariantNetwork()
inference_net = bf.networks.InvertibleNetwork(num_params=num_params)
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)


# If the checkpoint path does not exist, create it
checkpoint_path = f"checkpoint/{model_name}"

# We need to pass the custom configurator here
trainer = bf.trainers.Trainer(
    amortizer=amortizer, 
    generative_model=generative_model, 
    configurator=configurator,
    checkpoint_path=checkpoint_path)

# If there are already model fits:
# Make sure the checkpoint actually loads and DOES NOT SAY "Creating network from scratch"
# Instead it should say something like "Networks loaded from checkpoint/ckpt-1000"


# If the recovery plot path does not exist, create it
plot_path = f"recovery_plots/{model_name}"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)


if train_fitter:
    """Create validation simulations with some random N, if specific N is desired, need to 
    call simulator explicitly or define it with keyword arguments which can control behavior
    All trainer.train_*** can take additional keyword arguments controling the behavior of
    configurators, generative models and networks"""
    num_val = 300
    val_sims = generative_model(num_val)

    # Experience-replay training
    losses = trainer.train_experience_replay(epochs=num_epochs,
                                                 batch_size=32,
                                                 iterations_per_epoch=1000,
                                                 validation_sims=val_sims)
    # Validation, Loss Curves
    f = bf.diagnostics.plot_losses(losses['train_losses'], losses['val_losses'])
    f.savefig(f"{plot_path}/{model_name}_validation.png")
else:
    status = trainer.load_pretrained_network()



# Computational Adequacy
num_test = 12000
num_posterior_draws = 10000

# Need to test for different Ns, which is what the following code does
param_samples = np.empty((num_test, num_posterior_draws, num_params))
true_params = np.empty((num_test, num_params))
simulated_trial_nums = np.empty((num_test))
np.random.seed(2023) # Set the random seed to generate the same plots every time
for i in range(num_test):
    model_sims = configurator(generative_model(1))
    simulated_trial_nums[i] = model_sims['summary_conditions'].shape[1]
    true_params[i, :] = model_sims['parameters']
    param_samples[i, :, :] = amortizer.sample(model_sims, n_samples=num_posterior_draws)


print('For recovery plots, the mean number of simulated trials was %.0f +/- %.2f' %
    (np.mean(simulated_trial_nums), np.std(simulated_trial_nums)))


# BayesFlow native recovery plot, plot only up to 500 in each plot
fig = bf.diagnostics.plot_recovery(param_samples[0:500,:], true_params[0:500,:], param_names =
    ['drift', 'mu_boundary', 'beta', 'tau', 'var_boundary', 'dc', 'sigma1'])
fig.savefig(f"{plot_path}/{model_name}_true_vs_estimate.png")


# Posterior means
param_means = param_samples.mean(axis=1)

# Find the index of clearly good posterior means of tau (inside the prior range)
converged = (param_means[:, 3] > 0) & (param_means[:, 3] < 1)
print('%d of %d model fits were in the prior range for non-decision time' % 
    (np.sum(converged), converged.shape[0]))


# Plot true versus estimated for a subset of parameters
recovery_scatter(true_params[:, np.array([0, 5, 1, 2, 3])][0:500, :],
                  param_means[:, np.array([0, 5, 1, 2, 3])][0:500, :],
                  ['Drift Rate', 'Diffusion Coefficient', 'Boundary',
                  'Start Point', 'Non-Decision Time'],
                  font_size=16, color='#3182bdff', alpha=0.75, grantB1=False)
plt.savefig(f"{plot_path}/{model_name}_recovery_short.png")

# Calculate proportion of variance of external data explained by cognition
# var_eeg1 = var_alpha**2 + sigma1**2
data1_cognitive_var_samples = param_samples[:, :, 4]**2

true_data1_cognitive_var = true_params[:, 4]**2

data1_total_var_samples = data1_cognitive_var_samples + param_samples[:, :, 6]**2

true_data1_total_var = true_data1_cognitive_var + true_params[:, 6]**2

data1_cognitive_prop_samples = data1_cognitive_var_samples / data1_total_var_samples

true_data1_cognitive_prop = true_data1_cognitive_var / true_data1_total_var

# plt.figure()
# sns.kdeplot(true_data1_cognitive_prop)

# Plot the results
plt.figure()
# Use None to add singleton dimension for recovery which expects multiple chains
recovery(param_samples[0:500, :, 0, None],
    true_params[0:500, 0].squeeze())
plt.ylim(-5, 5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Drift')
plt.savefig(f'{plot_path}/{model_name}_Drift.png')
plt.close()

plt.figure()
recovery(param_samples[0:500, :, 1, None],
    true_params[0:500, 1].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Boundary')
plt.savefig(f'{plot_path}/{model_name}_Boundary.png')
plt.close()

plt.figure()
recovery(param_samples[0:500, :, 2, None],
    true_params[0:500, 2].squeeze())
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Relative Start Point')
plt.savefig(f'{plot_path}/{model_name}_StartPoint.png')
plt.close()

plt.figure()
recovery(param_samples[0:500, :, 3, None],
    true_params[0:500, 3].squeeze())
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Non-decision time')
plt.savefig(f'{plot_path}/{model_name}_NDT.png')
plt.close()

plt.figure()
recovery(param_samples[0:500, :, 4, None],
    true_params[0:500, 4].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Trial-to-trial variability in boundary')
plt.savefig(f'{plot_path}/{model_name}_boundary_variability.png')
plt.close()

plt.figure()
recovery(param_samples[0:500, :, 5, None],
    true_params[0:500, 5].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Diffusion coefficient')
plt.savefig(f'{plot_path}/{model_name}_DC.png')
plt.close()

plt.figure()
recovery(param_samples[0:500, :, 6, None],
    true_params[0:500, 6].squeeze())
plt.ylim(0.0, 6.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('data1 variance not related to cognition')
plt.savefig(f'{plot_path}/{model_name}_data1Noise.png')
plt.close()


plt.figure()
recovery(data1_cognitive_prop_samples[0:500, :, None],
    true_data1_cognitive_prop[0:500])
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Proportion data1 variance related to cognition')
plt.savefig(f'{plot_path}/{model_name}_data1prop_cog.png')
plt.close()


scatter_color = '#ABB0B8'

# Plot where the proportion of cognitive variance in the external data is large
cutoff_prop = .75
high_cog_sims = np.where((true_data1_cognitive_prop >= cutoff_prop))[0]

print('%d of %d model simulations had cognitive proportions above %.2f' % 
    (high_cog_sims.size, num_test, cutoff_prop))

# Plot the results
plt.figure()
# Use None to add singleton dimension for recovery which expects multiple chains
recovery(param_samples[high_cog_sims, :, 0, None],
    true_params[high_cog_sims, 0].squeeze())
plt.ylim(-5, 5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Drift')
plt.savefig(f'{plot_path}/{model_name}_Drift_high_prop.png')
plt.close()

plt.figure()
recovery(param_samples[high_cog_sims, :, 1, None],
    true_params[high_cog_sims, 1].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Boundary')
plt.savefig(f'{plot_path}/{model_name}_Boundary_high_prop.png')
plt.close()

plt.figure()
recovery(param_samples[high_cog_sims, :, 2, None],
    true_params[high_cog_sims, 2].squeeze())
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Relative Start Point')
plt.savefig(f'{plot_path}/{model_name}_StartPoint_high_prop.png')
plt.close()

plt.figure()
recovery(param_samples[high_cog_sims, :, 3, None],
    true_params[high_cog_sims, 3].squeeze())
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Non-decision time')
plt.savefig(f'{plot_path}/{model_name}_NDT_high_prop.png')
plt.close()

plt.figure()
recovery(param_samples[high_cog_sims, :, 4, None],
    true_params[high_cog_sims, 4].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Trial-to-trial variability in boundary')
plt.savefig(f'{plot_path}/{model_name}_boundary_variability_high_prop.png')
plt.close()

plt.figure()
recovery(param_samples[high_cog_sims, :, 5, None],
    true_params[high_cog_sims, 5].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Diffusion coefficient')
plt.savefig(f'{plot_path}/{model_name}_DC_high_prop.png')
plt.close()

plt.figure()
recovery(param_samples[high_cog_sims, :, 6, None],
    true_params[high_cog_sims, 6].squeeze())
plt.ylim(0.0, 6.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('data1 variance not related to cognition')
plt.savefig(f'{plot_path}/{model_name}_data1Noise_high_prop.png')
plt.close()


nplots = 18
plot_posterior2d(param_samples[high_cog_sims[0:nplots], :, 5].squeeze(),
    param_samples[high_cog_sims[0:nplots], :, 1].squeeze(),
   ['Diffusion coefficient', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_dc_high_prop.png")

plot_posterior2d(param_samples[high_cog_sims[0:nplots], :, 0].squeeze(),
    param_samples[high_cog_sims[0:nplots], :, 1].squeeze(),
   ['Drift rate', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_drift_high_prop.png")

plot_posterior2d(param_samples[high_cog_sims[0:nplots], :, 5].squeeze(),
    param_samples[high_cog_sims[0:nplots], :, 0].squeeze(),
   ['Diffusion coefficient', 'Drift rate'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_drift_dc_high_prop.png")

plot_posterior2d(param_samples[high_cog_sims[0:nplots], :, 3].squeeze(),
    param_samples[high_cog_sims[0:nplots], :, 0].squeeze(),
   ['Non-decision time', 'Drift rate'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_drift_ndt_high_prop.png")

plot_posterior2d(param_samples[high_cog_sims[0:nplots], :, 3].squeeze(),
    param_samples[high_cog_sims[0:nplots], :, 1].squeeze(),
   ['Non-decision time', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_ndt_high_prop.png")

plot_posterior2d(param_samples[high_cog_sims[0:nplots], :, 2].squeeze(),
    param_samples[high_cog_sims[0:nplots], :, 1].squeeze(),
   ['Start point', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_start_high_prop.png")

appendix_text = rf"""
The mean and standard deviation of number of simulated trials were $
{int(np.mean(simulated_trial_nums[high_cog_sims]))} \pm {int(np.std(simulated_trial_nums[high_cog_sims]))}$.
"""

print(appendix_text)

# Plot true versus estimated for a subset of parameters
recovery_scatter(true_params[:, np.array([0, 5, 1, 2, 3])][high_cog_sims, :],
                  param_means[:, np.array([0, 5, 1, 2, 3])][high_cog_sims, :],
                  ['Drift Rate', 'Diffusion Coefficient', 'Boundary',
                  'Start Point', 'Non-Decision Time'],
                  font_size=16, color='#3182bdff', alpha=0.75, grantB1=False)
plt.savefig(f"{plot_path}/{model_name}_recovery_short_high_prop.png")


# Plot where the proportion of cognitive variance in the external data is very large
cutoff_prop = .9
high_cog_sims = np.where((true_data1_cognitive_prop >= cutoff_prop))[0]

print('%d of %d model simulations had cognitive proportions above %.2f' % 
    (high_cog_sims.size, num_test, cutoff_prop))

# Plot the results
plt.figure()
# Use None to add singleton dimension for recovery which expects multiple chains
recovery(param_samples[high_cog_sims, :, 0, None],
    true_params[high_cog_sims, 0].squeeze())
plt.ylim(-5, 5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Drift')
plt.savefig(f'{plot_path}/{model_name}_Drift_higher_prop.png')
plt.close()

plt.figure()
recovery(param_samples[high_cog_sims, :, 1, None],
    true_params[high_cog_sims, 1].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Boundary')
plt.savefig(f'{plot_path}/{model_name}_Boundary_higher_prop.png')
plt.close()

plt.figure()
recovery(param_samples[high_cog_sims, :, 2, None],
    true_params[high_cog_sims, 2].squeeze())
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Relative Start Point')
plt.savefig(f'{plot_path}/{model_name}_StartPoint_higher_prop.png')
plt.close()

plt.figure()
recovery(param_samples[high_cog_sims, :, 3, None],
    true_params[high_cog_sims, 3].squeeze())
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Non-decision time')
plt.savefig(f'{plot_path}/{model_name}_NDT_higher_prop.png')
plt.close()

plt.figure()
recovery(param_samples[high_cog_sims, :, 4, None],
    true_params[high_cog_sims, 4].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Trial-to-trial variability in boundary')
plt.savefig(f'{plot_path}/{model_name}_boundary_variability_higher_prop.png')
plt.close()

plt.figure()
recovery(param_samples[high_cog_sims, :, 5, None],
    true_params[high_cog_sims, 5].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Diffusion coefficient')
plt.savefig(f'{plot_path}/{model_name}_DC_higher_prop.png')
plt.close()

plt.figure()
recovery(param_samples[high_cog_sims, :, 6, None],
    true_params[high_cog_sims, 6].squeeze())
plt.ylim(0.0, 6.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('data1 variance not related to cognition')
plt.savefig(f'{plot_path}/{model_name}_data1Noise_higher_prop.png')
plt.close()


nplots = 18
plot_posterior2d(param_samples[high_cog_sims[0:nplots], :, 5].squeeze(),
    param_samples[high_cog_sims[0:nplots], :, 1].squeeze(),
   ['Diffusion coefficient', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_dc_higher_prop.png")

plot_posterior2d(param_samples[high_cog_sims[0:nplots], :, 0].squeeze(),
    param_samples[high_cog_sims[0:nplots], :, 1].squeeze(),
   ['Drift rate', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_drift_higher_prop.png")

plot_posterior2d(param_samples[high_cog_sims[0:nplots], :, 5].squeeze(),
    param_samples[high_cog_sims[0:nplots], :, 0].squeeze(),
   ['Diffusion coefficient', 'Drift rate'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_drift_dc_higher_prop.png")

plot_posterior2d(param_samples[high_cog_sims[0:nplots], :, 3].squeeze(),
    param_samples[high_cog_sims[0:nplots], :, 0].squeeze(),
   ['Non-decision time', 'Drift rate'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_drift_ndt_higher_prop.png")

plot_posterior2d(param_samples[high_cog_sims[0:nplots], :, 3].squeeze(),
    param_samples[high_cog_sims[0:nplots], :, 1].squeeze(),
   ['Non-decision time', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_ndt_higher_prop.png")

plot_posterior2d(param_samples[high_cog_sims[0:nplots], :, 2].squeeze(),
    param_samples[high_cog_sims[0:nplots], :, 1].squeeze(),
   ['Start point', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_start_higher_prop.png")

appendix_text = rf"""
The mean and standard deviation of number of simulated trials were $
{int(np.mean(simulated_trial_nums[high_cog_sims]))} \pm {int(np.std(simulated_trial_nums[high_cog_sims]))}$.
"""

print(appendix_text)

# Plot true versus estimated for a subset of parameters
recovery_scatter(true_params[:, np.array([0, 5, 1, 2, 3])][high_cog_sims, :],
                  param_means[:, np.array([0, 5, 1, 2, 3])][high_cog_sims, :],
                  ['Drift Rate', 'Diffusion Coefficient', 'Boundary',
                  'Start Point', 'Non-Decision Time'],
                  font_size=16, color='#3182bdff', alpha=0.75, grantB1=False)
plt.savefig(f"{plot_path}/{model_name}_recovery_short_higher_prop.png")


# Plot a 3D joint posterior
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

main_color = '#332288'
secondary_color = '#ABB0B8'

# By default plot only one random posterior draws, draw 7
rand_draw = high_cog_sims[7]

# Main 3D scatter plot
ax.scatter(param_samples[rand_draw, :, 0].squeeze(),
           param_samples[rand_draw, :, 1].squeeze(),
           param_samples[rand_draw, :, 5].squeeze(), alpha=0.25, color=main_color)

# 2D scatter plot for drift rate and boundary (xy plane) at min diffusion coefficient
min_dc = param_samples[rand_draw, :, 5].min()
ax.scatter(param_samples[rand_draw, :, 0].squeeze(), param_samples[rand_draw, :, 1].squeeze(), 
    min_dc, alpha=0.25, color=secondary_color)

# 2D scatter plot for drift rate and diffusion coefficient (xz plane) at max boundary
max_boundary = param_samples[rand_draw, :, 1].max()
ax.scatter(param_samples[rand_draw, :, 0].squeeze(), max_boundary, 
    param_samples[rand_draw, :, 5].squeeze(), alpha=0.25, color=secondary_color)

# 2D scatter plot for boundary and diffusion coefficient (yz plane) at min drift rate
min_drift_rate = param_samples[rand_draw, :, 0].min()
ax.scatter(min_drift_rate, param_samples[rand_draw, :, 1].squeeze(), 
    param_samples[rand_draw, :, 5].squeeze(), alpha=0.25, color=secondary_color)

ax.set_xlabel(r'Drift rate ($\delta$)', fontsize=16, labelpad=10)
ax.set_ylabel(r'Boundary ($\alpha$)', fontsize=16, labelpad=10)
ax.set_zlabel(r'Diffusion coefficient ($\varsigma$)', fontsize=16, labelpad=10)

# Rotate the plot slightly clockwise around the z-axis
elevation = 20  # Default elevation
azimuth = -30   # Rotate 30 degrees counterclockwise from the default azimuth (which is -90)
ax.view_init(elev=elevation, azim=azimuth)

plt.savefig(f"{plot_path}/{model_name}_3d_posterior_drift_boundary_dc.png", dpi=300,
    bbox_inches="tight", pad_inches=0.5)

publication_text = rf"""
Draws from a joint posterior distribution for one simulated data set from Model dcDDM-$\alpha z$ 
(purple 3D scatter plot). Pairwise joint distributions are given by the grey projections on each 
of the three faces. The joint posterior distribution is driven mostly by the joint likelihood of 
the data (N={int(simulated_trial_nums[rand_draw])}) given the model.  The true 5-dimension joint posterior distribution also includes
the relative start point and non-decision time. The mean posteriors of those two parameters were 
$\hat\tau={np.mean(param_samples[rand_draw, :, 3]):.3}$ seconds and $\hat\beta={np.mean(param_samples[rand_draw, :, 2]):.2f}$ proportion of boundary in this simulation respectively.
"""


# Simulate a normal parameter space without measurement noise in EEG

# drift rate - index 0
drift = 3

# mean boundary - index 1
mu_alpha = 1.5

# relative start point - index 2
beta = .5

# non-decision time - index 3
ter = .4

# trial-to-trial variability in boundary - index 4
std_alpha = 1

# diffusion coefficient - index 5
dc = 1

# measurement noise of extdata1 - index 9
sigma1 = 0.1


sim_data1_cognitive_var = std_alpha**2
sim_data1_total_var = sim_data1_cognitive_var + sigma1**2
sim_data1_cognitive_prop = sim_data1_cognitive_var / sim_data1_total_var
print('The proportion of cognition explained by external data 1 is %.3f' % sim_data1_cognitive_prop)

input_params = np.hstack((drift, mu_alpha, beta, ter, std_alpha, dc, sigma1))

np.random.seed(2024) # Set the random seed to generate the same plots every time
n_trials = 300
obs_data = simulate_trials(input_params, n_trials)
obs_dict = {'sim_data': obs_data[np.newaxis,:,:], 
'sim_non_batchable_context': n_trials, 'prior_draws': input_params}
configured_dict = configurator(obs_dict)  # Make sure the data matches that configurator

# Obtain posterior samples
post_samples = amortizer.sample(configured_dict, num_posterior_draws)
plt.figure()
jellyfish(post_samples.T[:,:,None])
plt.savefig(f"{plot_path}/{model_name}_optimal_test_case.png")
print(f'The posterior means are {np.mean(post_samples,axis=0)}')