# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 20-March-23     Michael        Basic DDM with the diffusion coefficient equal to 1

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
from pyhddmjagsutils import recovery, recovery_scatter, plot_posterior2d


num_epochs = 500


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

    # drift ~ N(0, 2.0), mean drift rate
    drift = RNG.normal(0.0, 2.0)

    # alpha ~ N(1.0, 0.5) in [0, 10], boundary
    alpha = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=10)[0]

    # beta ~ Beta(2.0, 2.0), relative start point
    beta = RNG.beta(2.0, 2.0)

    # ter ~ N(0.5, 0.25) in [0, 1.5], non-decision time
    ter = truncnorm_better(mean=0.5, sd=0.25, low=0.0, upp=1.5)[0]

    p_samples = np.hstack((drift, alpha, beta, ter))
    return p_samples


num_params = draw_prior().shape[0]

@njit
def diffusion_trial(drift, boundary, beta, tau,
    dt=.01, max_steps=400.):
    """Simulates a trial from the diffusion model."""

    n_steps = 0.
    evidence = boundary * beta
   
  
    # Simulate a single DM path
    while ((evidence > 0) and (evidence < boundary) and (n_steps < max_steps)):

        # DDM equation
        evidence += drift*dt + np.sqrt(dt) * np.random.normal()

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

    drift, boundary, beta, tau = params
    rt = np.empty(n_trials)
    choice = np.empty(n_trials)
    for i in range(n_trials):
        rt[i], choice[i] = diffusion_trial(drift, boundary, beta, tau)

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


# If the recovery plot path does not exist, create it
plot_path = f"recovery_plots/{model_name}"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)


# Experience-replay training
losses = trainer.train_experience_replay(epochs=num_epochs,
                                             batch_size=32,
                                             iterations_per_epoch=1000,
                                             validation_sims=val_sims)
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
    ['drift', 'boundary', 'beta', 'tau'])
fig.savefig(f"{plot_path}/{model_name}_true_vs_estimate.png")


# Posterior means
param_means = param_samples.mean(axis=1)

# Find the index of clearly good posterior means of tau (inside the prior range)
converged = (param_means[:, 3] > 0) & (param_means[:, 3] < 1)
print('%d of %d model fits were in the prior range for non-decision time' % 
    (np.sum(converged), converged.shape[0]))


# Plot true versus estimated for a subset of parameters
recovery_scatter(true_params[:, np.array([0, 1, 2, 3])][:, :],
                  param_means[:, np.array([0, 1, 2, 3])][:, :],
                  ['Drift Rate', 'Boundary',
                  'Start Point', 'Non-Decision Time'],
                  font_size=16, color='#3182bdff', alpha=0.75, grantB1=False)
plt.savefig(f"{plot_path}/{model_name}_recovery_short.png")

# Plot the results
plt.figure()
# Use None to add singleton dimension for recovery which expects multiple chains
recovery(param_samples[:, :, 0, None],
    true_params[:, 0].squeeze())
plt.ylim(-6, 6)
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


# By default plot only the first 12 random posterior draws
plot_posterior2d(param_samples[0:12, :, 0].squeeze(),
    param_samples[0:12, :, 1].squeeze(),
   ['Drift rate', 'Boundary'],
   font_size=16, alpha=0.25)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_drift.png")

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