# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 17-March-23     Michael           Converted from single_trial_drift_dc_base.py

# References:
# https://github.com/stefanradev93/BayesFlow/blob/master/docs/source/tutorial_notebooks/LCA_Model_Posterior_Estimation.ipynb
# https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy

# To do:
# 1) Test if parameter standardization in configurator is useful

# Notes:
# 1) conda activate bf
# 2) Do not create checkpoint folder manually, 
# let BayesFlow do it otherwise get a no memory.pkl error

import os
import numpy as np
from scipy.stats import truncnorm
from numba import njit
import bayesflow as bf
import matplotlib.pyplot as plt
from pyhddmjagsutils import recovery, recovery_scatter

model_name = 'single_trial_drift_alpha_base'

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

    # alpha ~ N(1.0, 0.5) in [0, 10], boundary
    alpha = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=10)[0]

    # beta ~ Beta(2.0, 2.0), relative start point
    beta = RNG.beta(2.0, 2.0)

    # ter ~ N(0.5, 0.25) in [0, 1.5], non-decision time
    ter = truncnorm_better(mean=0.5, sd=0.25, low=0.0, upp=1.5)[0]

    # eta ~ N(1.0, 0.5) in [0, 3], trial-to-trial variability in drift
    eta = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=3)[0]

    # mu_dc ~ N(1.0, 0.5) in [0, 10], mean diffusion coefficient
    mu_dc = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=10)[0]

    # var_dc ~ N(1.0, 0.5) in [0, 3], trial-to-trial variability in diffusion coefficient
    var_dc = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=3)[0]

    p_samples = np.hstack((mu_drift, alpha, beta, ter, eta, mu_dc, var_dc))
    return p_samples


@njit
def diffusion_trial(mu_drift, boundary, beta, tau, eta, mu_dc, dc_var,
    dt=.01, max_steps=400.):
    """Simulates a trial from the diffusion model."""

    n_steps = 0.
    evidence = boundary * beta
   
    # trial-to-trial drift rate variability
    drift_trial = mu_drift + eta * np.random.normal()

    # trial-to-trial diffusion coefficient variability
    while True:
        dc_trial = mu_dc + dc_var * np.random.normal()
        if dc_trial>0:
            break
   
    # Simulate a single DM path
    while ((evidence > 0) and (evidence < boundary) and (n_steps < max_steps)):

        # DDM equation
        evidence += drift_trial*dt + np.sqrt(dt) * dc_trial * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt + tau

 
    if evidence >= boundary:
        choice =  1  # choice A
    elif evidence <= 0:
        choice = -1  # choice B
    else:
        choicert = 0  # This indicates a missing response
    return rt, choice

@njit
def simulate_trials(params, n_trials):
    """Simulates a diffusion process for trials ."""

    mu_drift, boundary, beta, tau, eta, mu_dc, dc_var = params
    rt = np.empty(n_trials)
    choice = np.empty(n_trials)
    for i in range(n_trials):
        rt[i], choice[i] = diffusion_trial(mu_drift, boundary, beta, tau, eta, mu_dc, dc_var)

    sim_data = np.stack((rt, choice), axis=-1)
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

# BayesFlow Setup
summary_net = bf.networks.InvariantNetwork()
inference_net = bf.networks.InvertibleNetwork(num_params=7)
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
losses = trainer.train_experience_replay(epochs=500,
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
# Need to test for different Ns, not just a random one!
num_test = 500
num_posterior_draws = 10000

np.random.seed(1)
model_sims = configurator(generative_model(num_test))
param_samples = amortizer.sample(model_sims, n_samples=num_posterior_draws)

true_params = model_sims['parameters']

# BayesFlow native recovery plot
fig = bf.diagnostics.plot_recovery(param_samples, true_params, param_names =
    ['mu_drift', 'boundary', 'beta', 'tau', 'eta', 'mu_dc', 'dc_var'])
fig.savefig(f"{plot_path}/{model_name}_true_vs_estimate.png")


# Posterior means
param_means = param_samples.mean(axis=1)

# Find the index of clearly good posterior means (inside the prior range)
# converged = (np.all(np.abs(param_means) < 5, axis=1))
converged = (np.all(np.abs(param_means) < 5, axis=1)) & (param_means[:, 3] > 0) & \
 (param_means[:, 3] < 1) & (param_means[:, 1] < 2)
print('%d of %d model fits were in the prior range for all parameters' % 
    (np.sum(converged), converged.shape[0]))


# Plot true versus estimated for a subset of parameters
recovery_scatter(true_params[:, np.array([0, 5, 1, 2, 3])][converged, :],
                  param_means[:, np.array([0, 5, 1, 2, 3])][converged, :],
                  ['Drift Rate', 'Diffusion Coefficient', 'Boundary',
                  'Start Point', 'Non-Decision Time'],
                  font_size=16, color='#3182bdff', alpha=0.75, grantB1=False)
plt.savefig(f"{plot_path}/{model_name}_recovery_short.png")
plt.savefig(f"{plot_path}/{model_name}_recovery_short.pdf")
plt.savefig(f"{plot_path}/{model_name}_recovery_short.svg")

# Plot the results
plt.figure()
# Use None to add singleton dimension for recovery which expects multiple chains
recovery(param_samples[converged, :, 0, None],
    true_params[converged, 0].squeeze())
plt.ylim(-5, 5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Drift')
plt.savefig(f'{plot_path}/{model_name}_Drift.png')
plt.close()

plt.figure()
recovery(param_samples[converged, :, 1, None],
    true_params[converged, 1].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Boundary')
plt.savefig(f'{plot_path}/{model_name}_Boundary.png')
plt.close()

plt.figure()
recovery(param_samples[converged, :, 2, None],
    true_params[converged, 2].squeeze())
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Relative Start Point')
plt.savefig(f'{plot_path}/{model_name}_StartPoint.png')
plt.close()

plt.figure()
recovery(param_samples[converged, :, 3, None],
    true_params[converged, 3].squeeze())
plt.ylim(0.0, 1.0)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Non-decision time')
plt.savefig(f'{plot_path}/{model_name}_NDT.png')
plt.close()

plt.figure()
recovery(param_samples[converged, :, 4, None],
    true_params[converged, 4].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Trial-to-trial variability in drift-rate')
plt.savefig(f'{plot_path}/{model_name}_drift_variability.png')
plt.close()

plt.figure()
recovery(param_samples[converged, :, 5, None],
    true_params[converged, 5].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Diffusion coefficient')
plt.savefig(f'{plot_path}/{model_name}_DC.png')
plt.close()

plt.figure()
recovery(param_samples[converged, :, 6, None],
    true_params[converged, 6].squeeze())
plt.ylim(0.0, 2.5)
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Diffusion coefficient variability')
plt.savefig(f'{plot_path}/{model_name}_DC_variability.png')
plt.close()