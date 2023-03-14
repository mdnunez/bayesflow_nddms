# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# ?            Amin Ghaderi                            Original code
# 13-Mar-23    Michael Nunez         Adaption for Snellius (Dutch supercomputing cluster)


import os
import numpy as np
from numba import njit
import bayesflow as bf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pyhddmjagsutils import recovery

model_name = 'NDDM_rel_ndt_bound_four_betas'


"""Generative Model Specifications 
User Defined Functions.""" 
def draw_prior():
    """Samples from the prior """
    p_samples = np.random.uniform(low =(-3.0, 0.1, 0.01, 0.05, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0),
                                  high=(3.0, 2.0, 0.99, 2.0,  0.5 , 1.0, 1.0, 1.0, 1.0, 1.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0))
    return p_samples

def prior_N(n_min=60, n_max=300):
    """A prior for the random number of observation"""
    return np.random.randint(n_min, n_max+1)

@njit
def diffusion_trial(delta, mu_alpha, beta, mu_tau, s_alpha, s_tau, sigma_1, sigma_2, sigma_3, sigma_4, gamma_11,  gamma_12,  gamma_21, gamma_22, gamma_31, gamma_32, gamma_41, gamma_42, dc=1.0, dt=.001):
    """Simulates a trial from the diffusion model."""

    n_steps = 0.
    
    # trial-to-trial boundary variability
    while True:
        alpha_trial = np.random.normal(mu_alpha, s_alpha)   
        if alpha_trial>0:
            break    
    # visual encoding time for each trial
    while True:
        tau_trial = np.random.normal(mu_tau, s_tau)
        if tau_trial>0:
            break  

    # starting evidence for each trial
    evidence = alpha_trial * beta
    
    # Simulate a single DM path
    while (evidence > 0 and evidence < alpha_trial):

        # DDM equation
        evidence += delta*dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt

    # b weights for each trial and each roi
    b = np.random.normal(gamma_11*alpha_trial + gamma_12*tau_trial, sigma_1)
    c = np.random.normal(gamma_21*alpha_trial + gamma_22*tau_trial, sigma_2)
    d = np.random.normal(gamma_31*alpha_trial + gamma_32*tau_trial, sigma_3)
    e = np.random.normal(gamma_41*alpha_trial + gamma_42*tau_trial, sigma_4)

    if evidence >= alpha_trial:
        choicert =  tau_trial + rt
    else:
        choicert = -tau_trial - rt
        
    return choicert, b, c, d, e

@njit
def simulate_trials(params, n_trials):
    """Simulates a diffusion process for trials ."""

    delta, mu_alpha, beta, mu_tau, s_alpha, s_tau, sigma_1, sigma_2, sigma_3, sigma_4, gamma_11,  gamma_12,  gamma_21, gamma_22, gamma_31, gamma_32, gamma_41, gamma_42 = params
    choicert = np.empty(n_trials)
    b = np.empty(n_trials)
    c = np.empty(n_trials)
    d = np.empty(n_trials)
    e = np.empty(n_trials)
    for i in range(n_trials):
        choicert[i], b[i], c[i], d[i], e[i] = diffusion_trial(delta, mu_alpha, beta, mu_tau, s_alpha, s_tau, sigma_1, sigma_2, sigma_3, sigma_4, gamma_11,  gamma_12,  gamma_21, gamma_22, gamma_31, gamma_32, gamma_41, gamma_42)  
    
    sim_data = np.stack((choicert, b, c, d, e), axis=-1)
    return sim_data

"""Connect via BayesFlow Wrappers
Note, that the same can be achieved using custom functions or classes, as long as the simulator and configurator interact well."""
prior = bf.simulation.Prior(prior_fun=draw_prior)
var_num_obs = bf.simulation.ContextGenerator(non_batchable_context_fun=prior_N)
simulator = bf.simulation.Simulator(simulator_fun=simulate_trials, context_generator=var_num_obs)
generative_model = bf.simulation.GenerativeModel(prior, simulator)


"""Create Configurator
We need this, since the variable N cannot be processed directly by the nets.""" 
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
inference_net = bf.networks.InvertibleNetwork(num_params=18)
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)

# If the checkpoint path does not exist, create it
checkpoint_path = f"checkpoint/{model_name}"

# We need to pass the custom configurator here
trainer = bf.trainers.Trainer(
    amortizer=amortizer, 
    generative_model=generative_model, 
    configurator=configurator,
    checkpoint_path=checkpoint_path)


"""Create validation simulations with some random N, if specific N is desired, need to 
call simulator explicitly or define it with keyword arguments which can control behavior
All trainer.train_*** can take additional keyword arguments controling the behavior of
configurators, generative models and networks"""
num_val = 300
val_sims = generative_model(num_val)

"""Quickcheck, var N is slow on my laptop, should definitely train longer for an actual application!"""
h = trainer.train_experience_replay(epochs=300, iterations_per_epoch=1000, batch_size=32, validation_sims=val_sims)

# If the recovery plot path does not exist, create it
plot_path = f"recovery_plots/{model_name}"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

"""Validation, Loss Curves"""
f = bf.diagnostics.plot_losses(h['train_losses'], h['val_losses'])
f.savefig(f"{plot_path}/{model_name}_validation.png")

"""Computational Adequacy"""
# Need to test for different Ns, not just a random one!
num_test = 500
num_posterior_draws_recovery = 1000
new_sims = configurator(generative_model(num_test))

posterior_draws = amortizer.sample(new_sims, n_samples=num_posterior_draws_recovery)
fig = bf.diagnostics.plot_recovery(posterior_draws, new_sims['parameters'], param_names = ['delta', 'mu_alpha', 'beta', 'mu_tau', 's_alpha', 's_tau', 'sigma_1', 'sigma_2', 'sigma_3', 'sigma_4', 'gamma_11',  'gamma_12',  'gamma_21', 'gamma_22', 'gamma_31', 'gamma_32', 'gamma_41', 'gamma_42'] )
fig.savefig(f"{plot_path}/{model_name}_true_vs_estimate.png")