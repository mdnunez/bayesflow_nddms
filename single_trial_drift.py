# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 06-Mar-23     Michael Nunez       Adapted from Amin's code here:
#https://github.com/AGhaderi/NDDM/blob/main/Single-trial-Integrative-CPP/Model7/CPP_single_trial_sigma_eta.py

import os
import numpy as np
from numba import njit
import bayesflow as bf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pyhddmjagsutils import recovery

model_name = 'single_trial_drift'

"""Using a better Estimated versus True parameter plot"""

def recovery_scatter(theta_true, theta_est, param_names,
                      figsize=(20, 4), font_size=12, color='blue', alpha=0.4,grantB1=False):
    """ Plots a scatter plot with abline of the estimated posterior means vs true values.

    Parameters
    ----------
    theta_true: np.array
        Array of true parameters.
    theta_est: np.array
        Array of estimated parameters.
    param_names: list(str)
        List of parameter names for plotting.
    dpi: int, default:300
        Dots per inch (dpi) for the plot.
    figsize: tuple(int, int), default: (20,4)
        Figure size.
    show: boolean, default: True
        Controls if the plot will be shown
    filename: str, default: None
        Filename if plot shall be saved
    font_size: int, default: 12
        Font size

    """


    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Determine n_subplots dynamically
    n_row = int(np.ceil(len(param_names) / 6))
    n_col = int(np.ceil(len(param_names) / n_row))

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat
        
    # --- Plot true vs estimated posterior means on a single row --- #
    for j in range(len(param_names)):
        
        # Plot analytic vs estimated
        axarr[j].scatter(theta_true[:, j], theta_est[:, j], color=color, alpha=alpha)
        
        # get axis limits and set equal x and y limits
        lower_lim = min(axarr[j].get_xlim()[0], axarr[j].get_ylim()[0])
        upper_lim = max(axarr[j].get_xlim()[1], axarr[j].get_ylim()[1])
        axarr[j].set_xlim((lower_lim, upper_lim))
        axarr[j].set_ylim((lower_lim, upper_lim))
        axarr[j].plot(axarr[j].get_xlim(), axarr[j].get_xlim(), '--', color='black')
        
        # Compute R2
        r2 = r2_score(theta_true[:, j], theta_est[:, j])
        axarr[j].text(0.1, 0.8, '$R^2$={:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes, 
                     size=font_size)
        
        axarr[j].set_xlabel('True %s' % param_names[j],fontsize=font_size)
        if j == 0:
            # Label plot
            axarr[j].set_ylabel('Estimated parameters',fontsize=font_size)
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)

        if grantB1:
            axarr[0].set_xlim(-4.5, 4.5)
            axarr[0].set_ylim(-4.5, 4.5)
            axarr[0].set_xticks([-4, -2, 0, 2, 4])
            axarr[0].set_yticks([-4, -2, 0, 2, 4])
            axarr[0].set_aspect('equal', adjustable='box')
            axarr[1].set_xlim(0.4, 2.1)
            axarr[1].set_ylim(0.4, 2.1)
            axarr[1].set_xticks([0.5, 1, 1.5, 2])
            axarr[1].set_yticks([0.5, 1, 1.5, 2])
            axarr[1].set_aspect('equal', adjustable='box')

    
    # Adjust spaces
    f.tight_layout()


# Generative Model Specifications User Defined Functions, non-batchable

def prior_N(n_min=60, n_max=300):
    """A prior for the random number of observation"""
    return np.random.randint(n_min, n_max+1)


def draw_prior():

    # Prior ranges for the simulator
    # mu_drift ~ U(0.01, 3.0)
    # boundary ~ U(0.5, 2.0)
    # beta ~ U(0.1, 0.9)  # relative start point
    # tau ~ U(0.1, 1.0)
    # sigma ~ U(0, 2)
    # Eta ~ U(0.0, 2.0)
    n_parameters = 6
    p_samples = np.random.uniform(low=(0.01,  0.5, 0.1, 0.1, 0.0, 0.0),
                                  high=(3.0, 2.0, 0.9, 1.0, 2.0,  2.0))
    return p_samples


@njit
def diffusion_trial(mu_drift, boundary, beta, tau, sigma, eta, dc=1.0, dt=.005):
    """Simulates a trial from the diffusion model."""

    n_steps = 0.
    evidence = boundary * beta
    
    # trial-to-trial drift rate variability
    drift_trial = mu_drift + eta * np.random.normal()
    
    # Simulate a single DM path
    while (evidence > 0 and evidence < boundary):

        # DDM equation
        evidence += drift_trial*dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt

    
    # EEG1
    eeg1 = np.random.normal(drift_trial, sigma)

    
    if evidence >= boundary:
        choicert =  tau + rt
        
    else:
        choicert = -tau - rt
    return choicert, eeg1


@njit
def simulate_trials(params, n_trials):
    """Simulates a diffusion process for trials ."""

    mu_drift, boundary, beta, tau, sigma, eta = params
    choicert = np.empty(n_trials)
    z = np.empty(n_trials)
    for i in range(n_trials):
        choicert[i], z[i] = diffusion_trial(mu_drift, boundary, beta, tau, sigma, eta)
    
    sim_data = np.stack((choicert, z), axis=-1)
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
inference_net = bf.networks.InvertibleNetwork(num_params=6)
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
    ['mu_drift', 'boundary', 'beta', 'tau', 'sigma', 'eta'])
fig.savefig(f"{plot_path}/{model_name}_true_vs_estimate.png")

