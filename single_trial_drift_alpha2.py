# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 20-March-23     Michael Nunez      Conversion from single_trial_drift_alpha.py
#                  predicts standardized EEG measures,
#    increased prior range for noise in EEG to allow for models without EEG relationships

# References:
# https://github.com/stefanradev93/BayesFlow/blob/master/docs/source/tutorial_notebooks/LCA_Model_Posterior_Estimation.ipynb
# https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy

# To do:
# 1) Test if parameter standardization in configurator is useful

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
from pyhddmjagsutils import recovery, recovery_scatter, plot_posterior2d

num_epochs = 1
view_simulation = False

# Get the filename of the currently running script
filename = os.path.basename(__file__)

# Remove the .py extension from the filename
model_name = os.path.splitext(filename)[0]

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

    # mu_drift ~ N(0, 2.0), mean drift rate
    mu_drift = RNG.normal(0.0, 2.0)

    # mu_alpha ~ N(1.0, 0.5) in [0, 10], mean boundary
    mu_alpha = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=10)[0]

    # beta ~ Beta(2.0, 2.0), relative start point
    beta = RNG.beta(2.0, 2.0)

    # ter ~ N(0.5, 0.25) in [0, 1.5], non-decision time
    ter = truncnorm_better(mean=0.5, sd=0.25, low=0.0, upp=1.5)[0]

    # eta ~ N(1.0, 0.5) in [0, 3], trial-to-trial variability in drift
    eta = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=3)[0]

    # dc ~ N(1.0, 0.5) in [0, 10], mean diffusion coefficient
    dc = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=10)[0]

    # var_alpha ~ N(1.0, 0.5) in [0, 3], trial-to-trial variability in boundary
    var_alpha = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=3)[0]

    # gamma_dr1 = 1 Fixed effect of single-trial drift rate on EEG1

    # gamma_bd1 ~ N(0, 1.0), Effect of single-trial boundary on EEG1
    gamma_dc1 = RNG.normal(0.0, 1.0)

    # gamma_dr2 ~ N(0, 1.0), Effect of single-trial drift rate on EEG2
    gamma_dr2 = RNG.normal(0.0, 1.0)

    # gamma_bd2 = 1, Fixed effect of single-trial boundary on EEG2

    # sigma1 ~ U(0.0, 1.0),measurement noise of EEG1, less than 1 (assume standardized measure)
    sigma1 = RNG.uniform(0.0, 5.0)

    # sigma2 ~ U(0.0, 1.0),measurement noise of EEG2, less than 1 (assume standardized measure)
    sigma2 = RNG.uniform(0.0, 5.0)

    p_samples = np.hstack((mu_drift, mu_alpha, beta, ter, eta, dc, var_alpha, gamma_dc1,
        gamma_dr2, sigma1, sigma2))
    return p_samples


num_params = draw_prior().shape[0]

@njit
def diffusion_trial(mu_drift, mu_alpha, beta, tau, eta, dc, var_alpha,
	gamma_bd1, gamma_dr2, sigma1, sigma2, dt=.01, max_steps=400.):
    """Simulates a trial from the diffusion model."""

    # trial-to-trial drift rate variability
    drift_trial = mu_drift + eta * np.random.normal()

    # trial-to-trial diffusion coefficient variability
    while True:
        bound_trial = mu_alpha + var_alpha * np.random.normal()
        if bound_trial>0:
            break

    n_steps = 0.
    evidence = bound_trial * beta
  
    # Simulate a single DM path
    while ((evidence > 0) and (evidence < bound_trial) and (n_steps < max_steps)):

        # DDM equation
        evidence += drift_trial*dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt

   
    # EEG1
    temp_eeg1 = np.random.normal(1*drift_trial + gamma_bd1*bound_trial, sigma1)
    # Observe only standardized measures
    mu_eeg1 = 1*mu_drift + gamma_bd1*mu_alpha
    var_eeg1 = eta**2 + (gamma_bd1**2 * var_alpha**2) + sigma1**2
    eeg1 = (temp_eeg1 - mu_eeg1) / np.sqrt(var_eeg1)

    # EEG2
    temp_eeg2 = np.random.normal(gamma_dr2*drift_trial + 1*bound_trial, sigma2)
    # Observe only standardized measures
    mu_eeg2 = gamma_dr2*mu_drift + mu_alpha
    var_eeg2 = (gamma_dr2**2 * eta**2) + var_alpha**2 + sigma2**2
    eeg2 = (temp_eeg2 - mu_eeg2) / np.sqrt(var_eeg2)

  
    if evidence >= bound_trial:
        choicert =  tau + rt  
    elif evidence <= 0:
        choicert = -tau - rt
    else:
        choicert = 0  # This indicates a missing response
    return choicert, eeg1, eeg2

@njit
def simulate_trials(params, n_trials):
    """Simulates a diffusion process for trials ."""

    mu_drift, mu_alpha, beta, tau, eta, dc, var_alpha, gamma_bd1, gamma_dr2, sigma1, sigma2 = params
    choicert = np.empty(n_trials)
    z1 = np.empty(n_trials)
    z2 = np.empty(n_trials)
    for i in range(n_trials):
        choicert[i], z1[i], z2[i] = diffusion_trial(mu_drift, mu_alpha, beta, tau, eta, dc, var_alpha, gamma_bd1, gamma_dr2, sigma1, sigma2)
   
    sim_data = np.stack((choicert, z1, z2), axis=-1)
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
    eeg1_means =np.empty((num_test))
    eeg1_vars = np.empty((num_test))
    eeg2_means =np.empty((num_test))
    eeg2_vars = np.empty((num_test))
    rt_means = np.empty((num_test))
    choice_means = np.empty((num_test))
    np.random.seed(2023) # Set the random seed to generate the same plots every time
    for i in range(num_test):
        raw_sims = generative_model(1)
        these_sims = raw_sims['sim_data']
        eeg1_means[i] = np.mean(np.squeeze(these_sims[0, :, 1]))
        eeg1_vars[i] = np.var(np.squeeze(these_sims[0, :, 1]))
        eeg2_means[i] = np.mean(np.squeeze(these_sims[0, :, 2]))
        eeg2_vars[i] = np.var(np.squeeze(these_sims[0, :, 2]))
        rt_means[i] = np.mean(np.abs(np.squeeze(these_sims[0,:, 0])))
        choice_means[i] = np.mean(.5 + .5*np.sign(np.squeeze(these_sims[0,:, 0]))) #convert [1, -1] to [1, 0]


    # This should include a large mass around 0
    plt.figure()
    sns.kdeplot(eeg1_means)

    # This should include a large mass around 1
    plt.figure()
    sns.kdeplot(eeg1_vars)

    # This should include a large mass around 0
    plt.figure()
    sns.kdeplot(eeg2_means)

    # This should include a large mass around 1
    plt.figure()
    sns.kdeplot(eeg2_vars)

    plt.figure()
    sns.kdeplot(rt_means)

    plt.figure()
    sns.kdeplot(choice_means)

    # This should usually be standard normal
    plt.figure()
    sns.kdeplot(np.squeeze(these_sims[0, :, 1]))

    # This should usually be standard normal
    plt.figure()
    sns.kdeplot(np.squeeze(these_sims[0, :, 2]))


    sim_rts = np.abs(np.squeeze(these_sims[0, :, 0]))
    sim_choices = np.sign(np.squeeze(these_sims[0,:, 0]))
    # This should look like a shifted Wald
    plt.figure()
    sns.kdeplot(sim_rts[sim_choices == 1])


    # This should look like a shifted Wald
    plt.figure()
    sns.kdeplot(sim_rts[sim_choices == -1])


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


# If the recovery plot path does not exist, create it
plot_path = f"recovery_plots/{model_name}"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Validation, Loss Curves
f = bf.diagnostics.plot_losses(losses['train_losses'], losses['val_losses'])
f.savefig(f"{plot_path}/{model_name}_validation.png")

# Computational Adequacy
num_test = 500
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


# BayesFlow native recovery plot
fig = bf.diagnostics.plot_recovery(param_samples, true_params, param_names =
    ['mu_drift', 'mu_boundary', 'beta', 'tau', 'eta', 'dc', 'var_boundary', 'gamma_bd1',
    'gamma_dr2', 'sigma1', 'sigma2'])
fig.savefig(f"{plot_path}/{model_name}_true_vs_estimate.png")


# Posterior means
param_means = param_samples.mean(axis=1)

# Find the index of clearly good posterior means of tau (inside the prior range)
converged = (param_means[:, 3] > 0) & (param_means[:, 3] < 1)
print('%d of %d model fits were in the prior range for non-decision time' % 
    (np.sum(converged), converged.shape[0]))


# Plot true versus estimated for a subset of parameters
recovery_scatter(true_params[:, np.array([0, 5, 1, 2, 3])][:, :],
                  param_means[:, np.array([0, 5, 1, 2, 3])][:, :],
                  ['Drift Rate', 'Diffusion Coefficient', 'Boundary',
                  'Start Point', 'Non-Decision Time'],
                  font_size=16, color='#3182bdff', alpha=0.75, grantB1=False)
plt.savefig(f"{plot_path}/{model_name}_recovery_short.png")

# Plot the results
plt.figure()
# Use None to add singleton dimension for recovery which expects multiple chains
recovery(param_samples[:, :, 0, None],
    true_params[:, 0].squeeze())
plt.ylim(-5, 5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Drift')
plt.savefig(f'{plot_path}/{model_name}_Drift.png')
plt.close()

plt.figure()
recovery(param_samples[:, :, 1, None],
    true_params[:, 1].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Boundary')
plt.savefig(f'{plot_path}/{model_name}_Boundary.png')
plt.close()

plt.figure()
recovery(param_samples[:, :, 2, None],
    true_params[:, 2].squeeze())
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Relative Start Point')
plt.savefig(f'{plot_path}/{model_name}_StartPoint.png')
plt.close()

plt.figure()
recovery(param_samples[:, :, 3, None],
    true_params[:, 3].squeeze())
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Non-decision time')
plt.savefig(f'{plot_path}/{model_name}_NDT.png')
plt.close()

plt.figure()
recovery(param_samples[:, :, 4, None],
    true_params[:, 4].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Trial-to-trial variability in drift-rate')
plt.savefig(f'{plot_path}/{model_name}_drift_variability.png')
plt.close()

plt.figure()
recovery(param_samples[:, :, 5, None],
    true_params[:, 5].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Diffusion coefficient')
plt.savefig(f'{plot_path}/{model_name}_DC.png')
plt.close()

plt.figure()
recovery(param_samples[:, :, 6, None],
    true_params[:, 6].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Trial-to-trial boundary variability')
plt.savefig(f'{plot_path}/{model_name}_Boundary_variability.png')
plt.close()


plt.figure()
recovery(param_samples[:, :, 7, None],
    true_params[:, 7].squeeze())
plt.ylim(-3.0, 3.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Effect of boundary on EEG1')
plt.savefig(f'{plot_path}/{model_name}_Effect_of_boundary_EEG1.png')
plt.close()

plt.figure()
recovery(param_samples[:, :, 8, None],
    true_params[:, 8].squeeze())
plt.ylim(-3.0, 3.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Effect of drift on EEG2')
plt.savefig(f'{plot_path}/{model_name}_Effect_of_drift_EEG2.png')
plt.close()

plt.figure()
recovery(param_samples[:, :, 9, None],
    true_params[:, 9].squeeze())
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('EEG1 variance not related to cognition')
plt.savefig(f'{plot_path}/{model_name}_EEG1Noise.png')
plt.close()

plt.figure()
recovery(param_samples[:, :, 10, None],
    true_params[:, 10].squeeze())
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('EEG2 variance not related to cognition')
plt.savefig(f'{plot_path}/{model_name}_EEG2Noise.png')
plt.close()

# By default plot only the first 12 random posterior draws
plot_posterior2d(param_samples[0:12, :, 5].squeeze(),
    param_samples[0:12, :, 1].squeeze(),
   ['Diffusion coefficient', 'Boundary'],
   font_size=16, alpha=0.25)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_dc.png")

plot_posterior2d(param_samples[0:12, :, 0].squeeze(),
    param_samples[0:12, :, 1].squeeze(),
   ['Drift rate', 'Boundary'],
   font_size=16, alpha=0.25)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_drift.png")

plot_posterior2d(param_samples[0:12, :, 5].squeeze(),
    param_samples[0:12, :, 0].squeeze(),
   ['Diffusion coefficient', 'Drift rate'],
   font_size=16, alpha=0.25)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_drift_dc.png")

plot_posterior2d(param_samples[0:12, :, 3].squeeze(),
    param_samples[0:12, :, 0].squeeze(),
   ['Non-decision time', 'Drift rate'],
   font_size=16, alpha=0.25)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_drift_ndt.png")

plot_posterior2d(param_samples[0:12, :, 3].squeeze(),
    param_samples[0:12, :, 1].squeeze(),
   ['Non-decision time', 'Boundary'],
   font_size=16, alpha=0.25)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_ndt.png")

plot_posterior2d(param_samples[0:12, :, 2].squeeze(),
    param_samples[0:12, :, 1].squeeze(),
   ['Start point', 'Boundary'],
   font_size=16, alpha=0.25)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_start.png")