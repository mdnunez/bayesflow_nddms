# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 14-Feb-2024   Michael D. Nunez      Converted from single_trial_alpha_fixed.py

# UNFINISHED

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
train_fitter = True
make_recovery_plots = True


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

    # std_alpha ~ N(1.0, 0.5) in [0, 3], trial-to-trial std in boundary, index 4
    std_alpha = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=3)[0]

    #dc ~ N(1.0, 0.5) in [0, 10], diffusion coefficient, index 5
    dc = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=10)[0]

    # sigma1 ~ U(0.0, 5.0),measurement noise of extdata1, index 6
    sigma1 = RNG.uniform(0.0, 5.0)

    # sigma2 ~ U(0.0, 5.0),measurement noise of single-trial boundary, index 7
    sigma2 = RNG.uniform(0.0, 5.0)

    p_samples = np.hstack((drift, mu_alpha, beta, ter, std_alpha, dc, sigma1, 
        sigma2))
    return p_samples


num_params = draw_prior().shape[0]

@njit
def diffusion_trial(drift, mu_alpha, beta, ter, std_alpha, dc, sigma1, sigma2 
    dt=.01, max_steps=400.):
    """Simulates a trial from the diffusion model."""

    # shared trial-to-trial parameter
    bound_trial = mu_alpha + std_alpha * np.random.normal()

    # trial-to-trial boundary
    while True:
        obs_bound = np.random.normal(1*bound_trial, sigma2)
        if obs_bound>0:
            break

    n_steps = 0.
    evidence = obs_bound * beta
 
    # Simulate a single DM path
    while ((evidence > 0) and (evidence < obs_bound) and (n_steps < max_steps)):

        # DDM equation
        evidence += drift*dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt

 
    # Observe absolute measures with noise
    extdata1 = np.random.normal(1*bound_trial, sigma1)

    if evidence >= obs_bound:
        choicert =  ter + rt  
    elif evidence <= 0:
        choicert = -ter - rt
    else:
        choicert = 0  # This indicates a missing response
    return choicert, extdata1

@njit
def simulate_trials(params, n_trials):
    """Simulates a diffusion process for trials ."""

    drift, mu_alpha, beta, ter, std_alpha, dc, sigma1, sigma2 = params
    choicert = np.empty(n_trials)
    z1 = np.empty(n_trials)
    for i in range(n_trials):
        choicert[i], z1[i] = diffusion_trial(drift, mu_alpha, beta, ter, 
            std_alpha, dc, sigma1, sigma2)
   
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


# Define checkpoint path
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

