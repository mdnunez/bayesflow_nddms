# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 12-May-2023   Michael Nunez   Converted from basic_ddm_dc_evidence_trainlow
#    This model assumes that the process is observed without noise, but scaled

# References:
# https://github.com/stefanradev93/BayesFlow/blob/master/docs/source/tutorial_notebooks/LCA_Model_Posterior_Estimation.ipynb
# https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
# https://docs.python.org/3/library/timeit.html

# To do:
# 1) Test if parameter standardization in configurator is useful

# Notes:
# 1) conda activate bf
# 2) Do not create checkpoint folder manually, 
# let BayesFlow do it otherwise get a no memory.pkl error

# Paper reference:
# Model dcDDM in the manuscript: Nunez, Schubert, Frischkorn, Oberauer 2023.

import os
import numpy as np
from scipy.stats import truncnorm
from numba import njit
import seaborn as sns
import bayesflow as bf
import matplotlib.pyplot as plt
from pyhddmjagsutils import recovery, recovery_scatter, plot_posterior2d

train_fitter = True
num_epochs = 500
view_simulation = False


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

    # drift ~ N(0, 2.0), mean drift rate
    drift = RNG.normal(0.0, 2.0)

    # alpha ~ N(1.0, 0.5) in [0, 10], boundary
    alpha = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=10)[0]

    # beta ~ Beta(2.0, 2.0), relative start point
    beta = RNG.beta(2.0, 2.0)

    # ter ~ N(0.5, 0.25) in [0, 1.5], non-decision time
    ter = truncnorm_better(mean=0.5, sd=0.25, low=0.0, upp=1.5)[0]

    # dc ~ N(1.0, 0.5) in [0, 10], diffusion coefficient
    dc = truncnorm_better(mean=1.0, sd=0.5, low=0.0, upp=10)[0]

    p_samples = np.hstack((drift, alpha, beta, ter, dc))
    return p_samples


num_params = draw_prior().shape[0]

@njit
def diffusion_trial(drift=3, boundary=1, beta=.5, tau=.4, dc=1,
    dt=.001, max_time=4.):
    """Simulates a trial from the diffusion model."""

    max_steps = max_time/dt

    n_steps = 0.
    evidence = boundary * beta

    # Assume noisey correlate of the evidence time course is observed
    n_eeg_obs = int(.2/dt)
    erp_path = np.zeros((n_eeg_obs))
  
 
    # Simulate a single DM path
    while ((evidence > 0) and (evidence < boundary) and (n_steps < max_steps)):

        # DDM equation
        evidence += drift*dt + np.sqrt(dt) * dc * np.random.normal()

        # Assume noisey correlate of the evidence time course 
        #is observed for 200 ms at the beginning of an evidence path
        if n_steps < n_eeg_obs:
            erp_path[int(n_steps)] = evidence

        # Increment step
        n_steps += 1.0

    # Calculate response time
    rt = n_steps * dt + tau

    # Assume evidence time course stays at the boundary, strong assumption
    erp_path[int(n_steps):] = evidence

    # Add small noise to observed paths only for computational reasons
    # E.g avoid 0s given by np.std() below when evidence path quickly reaches boundary
    erp_path = erp_path + np.random.normal(0, 0.001, size=n_eeg_obs)

    # Standardize the observed EEG path
    obs_path = (erp_path - np.mean(erp_path))/np.std(erp_path)


    if evidence >= boundary:
        choice =  1  # choice A
    elif evidence <= 0:
        choice = -1  # choice B
    else:
        choice = 0  # This indicates a missing response
    return rt, choice, obs_path



@njit
def simulate_trials(params, n_trials):
    """Simulates a diffusion process for trials ."""

    drift, boundary, beta, tau, dc = params
    rt = np.empty((n_trials, 1))
    choice = np.empty((n_trials, 1))
    obs_path = np.empty((n_trials, 200))
    for i in range(n_trials):
        rt[i], choice[i], obs_path[i, :] = diffusion_trial(drift, boundary, beta, tau, dc)

    sim_data = np.concatenate((rt, choice, obs_path), axis=1)
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
    # Plot the distributions of choice RTs and EEG data
    num_test = 50
    noise_cutoff = .5


    # Need to test for different Ns, which is what the following code does
    erps =np.empty((num_test,200))
    rt_means = np.empty((num_test))
    choice_means = np.empty((num_test))
    np.random.seed(2023) # Set the random seed to generate the same plots every time
    for i in range(num_test):
        raw_sims = generative_model(1)
        these_sims = raw_sims['sim_data']
        erps[i, :] = np.mean(np.squeeze(these_sims[0, :, 2:]),axis=0)
        rt_means[i] = np.mean(np.abs(np.squeeze(these_sims[0,:, 0])))
        choice_means[i] = np.mean(.5 + .5*np.squeeze(these_sims[0,:, 1])) #convert [1, -1] to [1, 0]

    plt.figure()
    plt.plot(np.linspace(0, .2,num=200),erps.T)
    plt.xlabel('Time (sec)')
    plt.ylabel('Standardized microvolts')

    plt.figure()
    sns.kdeplot(rt_means)

    plt.figure()
    sns.kdeplot(choice_means)

    sim_rts = np.abs(np.squeeze(these_sims[0, :, 0]))
    sim_choices = np.sign(np.squeeze(these_sims[0,:, 0]))
    sim_evidence = np.squeeze(these_sims[0,:,2:])
    # This are the evidence paths from the final simulation
    plt.figure()
    plt.plot(np.linspace(0, .2,num=200),sim_evidence.T)

    # This should look like a shifted Wald
    plt.figure()
    sns.kdeplot(sim_rts[sim_choices == 1])


    # This should look like a shifted Wald
    plt.figure()
    sns.kdeplot(sim_rts[sim_choices == -1])

    plt.show()

    # Minimum RT
    print('The minimum RT is %.3f when the NDT was %.3f' % 
        (np.min(sim_rts[sim_rts != 0]), raw_sims['prior_draws'][0,3]))




# BayesFlow Setup
summary_net = bf.networks.InvariantNetwork()
inference_net = bf.networks.InvertibleNetwork(num_params=num_params)
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)


# Let BayesFlow create the checkpoint path to avoid the no memory.pkl error
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
num_test = 10000
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
    ['drift', 'boundary', 'beta', 'tau', 'dc'])
fig.savefig(f"{plot_path}/{model_name}_true_vs_estimate.png")


# Posterior means
param_means = param_samples.mean(axis=1)

# Find the index of clearly good posterior means of tau (inside the prior range)
converged = (param_means[:, 3] > 0) & (param_means[:, 3] < 1)
print('%d of %d model fits were in the prior range for non-decision time' % 
    (np.sum(converged), converged.shape[0]))


# Plot true versus estimated for a subset of parameters
recovery_scatter(true_params[:, np.array([0, 4, 1, 2, 3])][0:500, :],
                  param_means[:, np.array([0, 4, 1, 2, 3])][0:500, :],
                  ['Drift Rate', 'Diffusion Coefficient', 'Boundary',
                  'Start Point', 'Non-Decision Time'],
                  font_size=16, color='#3182bdff', alpha=0.75, grantB1=False)
plt.savefig(f"{plot_path}/{model_name}_recovery_short.png")

# Plot the results
plt.figure()
# Use None to add singleton dimension for recovery which expects multiple chains
recovery(param_samples[0:500, :, 0, None],
    true_params[0:500, 0].squeeze())
plt.ylim(-6, 6)
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
plt.title('Diffusion coefficient')
plt.savefig(f'{plot_path}/{model_name}_DC.png')
plt.close()

scatter_color = '#ABB0B8'

# By default plot only the first 18 random posterior draws
nplots = 18
plot_posterior2d(param_samples[0:nplots, :, 4].squeeze(),
    param_samples[0:nplots, :, 1].squeeze(),
   ['Diffusion coefficient', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_dc.png")

plot_posterior2d(param_samples[0:nplots, :, 0].squeeze(),
    param_samples[0:nplots, :, 1].squeeze(),
   ['Drift rate', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_drift.png")

plot_posterior2d(param_samples[0:nplots, :, 4].squeeze(),
    param_samples[0:nplots, :, 0].squeeze(),
   ['Diffusion coefficient', 'Drift rate'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_drift_dc.png")

plot_posterior2d(param_samples[0:nplots, :, 3].squeeze(),
    param_samples[0:nplots, :, 0].squeeze(),
   ['Non-decision time', 'Drift rate'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_drift_ndt.png")

plot_posterior2d(param_samples[0:nplots, :, 3].squeeze(),
    param_samples[0:nplots, :, 1].squeeze(),
   ['Non-decision time', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_ndt.png")

plot_posterior2d(param_samples[0:nplots, :, 2].squeeze(),
    param_samples[0:nplots, :, 1].squeeze(),
   ['Start point', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_start.png")

appendix_text = rf"""
The mean and standard deviation of number of simulated trials were $
{int(np.mean(simulated_trial_nums[0:nplots]))} \pm {int(np.std(simulated_trial_nums[0:nplots]))}$.
"""

print(appendix_text)

# Plot a 3D joint posterior
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

main_color = '#332288'
secondary_color = '#ABB0B8'

# By default plot only one random posterior draws, draw 7
rand_draw = 13

# Main 3D scatter plot
ax.scatter(param_samples[rand_draw, :, 0].squeeze(),
           param_samples[rand_draw, :, 1].squeeze(),
           param_samples[rand_draw, :, 4].squeeze(), alpha=0.25, color=main_color)

# 2D scatter plot for drift rate and boundary (xy plane) at min diffusion coefficient
min_dc = param_samples[rand_draw, :, 4].min()
ax.scatter(param_samples[rand_draw, :, 0].squeeze(), param_samples[rand_draw, :, 1].squeeze(), 
    min_dc, alpha=0.25, color=secondary_color)

# 2D scatter plot for drift rate and diffusion coefficient (xz plane) at max boundary
max_boundary = param_samples[rand_draw, :, 1].max()
ax.scatter(param_samples[rand_draw, :, 0].squeeze(), max_boundary, 
    param_samples[rand_draw, :, 4].squeeze(), alpha=0.25, color=secondary_color)

# 2D scatter plot for boundary and diffusion coefficient (yz plane) at min drift rate
min_drift_rate = param_samples[rand_draw, :, 0].min()
ax.scatter(min_drift_rate, param_samples[rand_draw, :, 1].squeeze(), 
    param_samples[rand_draw, :, 4].squeeze(), alpha=0.25, color=secondary_color)

ax.set_xlabel(r'Drift rate ($\delta$)', fontsize=16, labelpad=10)
ax.set_ylabel(r'Boundary ($\alpha$)', fontsize=16, labelpad=10)
ax.set_zlabel(r'Diffusion coefficient ($\varsigma$)', fontsize=16, labelpad=10)

# Rotate the plot slightly clockwise around the z-axis
elevation = 20  # Default elevation
azimuth = -30   # Rotate 30 degrees counterclockwise from the default azimuth (which is -90)
ax.view_init(elev=elevation, azim=azimuth)

plt.savefig(f"{plot_path}/{model_name}_3d_posterior_drift_boundary_dc.png", dpi=300,
    bbox_inches="tight", pad_inches=0.5)

