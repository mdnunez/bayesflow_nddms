# basic_ddm_dc_pystan2.py - Testing JAGS fits of a non-hierarchical DDM model 
#               with diffusion coefficient in Stan using pystan 2 in Python 3
#
# Copyright (C) 2024 Michael D. Nunez, <m.d.nunez@uva.nl>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 25/09/23      Michael Nunez      Original code priors match basic_ddm_dc.py
# 03/10/23      Michael Nunez                   Updates to work with Pystan 2
# 13/03/24      Michael Nunez    Create empty directories if they do not exist
# 21/08/24      Michael Nunez       Fix bug in Stan code (see pull request)          


# Online references
# https://pystan.readthedocs.io/en/latest/index.html vs 
# https://pystan2.readthedocs.io/en/latest/getting_started.html


# Modules
import numpy as np
import pystan
import scipy.io as sio
from scipy import stats
import warnings
import os
import matplotlib.pyplot as plt
import pyhddmjagsutils as phju


### Simulations ###

data_path = f"data"
if not os.path.exists(data_path):
    os.makedirs(data_path)

if not os.path.exists('data/basic_ddm_dc_test1.mat'):
    # Number of simulated participants
    nparts = 100

    # Number of trials for one participant
    ntrials = 100

    # Number of total trials in each simulation
    N = ntrials * nparts

    # Set random seed
    np.random.seed(2021)

    ndt = np.random.uniform(.15, .6, size=nparts)  # Uniform from .15 to .6 seconds
    alpha = np.random.uniform(.8, 1.4, size=nparts)  # Uniform from .8 to 1.4 evidence units
    beta = np.random.uniform(.3, .7, size=nparts)  # Uniform from .3 to .7 * alpha
    delta = np.random.uniform(-4, 4, size=nparts)  # Uniform from -4 to 4 evidence units per second
    varsigma = np.random.uniform(.8, 1.4, size=nparts) # Uniform from .8 to 1.4 evidence units
    deltatrialsd = np.random.uniform(0, 2, size=nparts)  # Uniform from 0 to 2 evidence units per second
    y = np.zeros(N)
    rt = np.zeros(N)
    acc = np.zeros(N)
    participant = np.zeros(N)  # Participant index
    indextrack = np.arange(ntrials)
    for p in range(nparts):
        tempout = phju.simulratcliff(N=ntrials, Alpha=alpha[p], Tau=ndt[p], Beta=beta[p],
                                     Nu=delta[p], Eta=deltatrialsd[p], Varsigma=varsigma[p])
        tempx = np.sign(np.real(tempout))
        tempt = np.abs(np.real(tempout))
        y[indextrack] = tempx * tempt
        rt[indextrack] = tempt
        acc[indextrack] = (tempx + 1) / 2
        participant[indextrack] = p + 1
        indextrack += ntrials

    genparam = dict()
    genparam['ndt'] = ndt
    genparam['beta'] = beta
    genparam['alpha'] = alpha
    genparam['delta'] = delta
    genparam['deltatrialsd'] = deltatrialsd
    genparam['varsigma'] = varsigma
    genparam['rt'] = rt
    genparam['acc'] = acc
    genparam['y'] = y
    genparam['participant'] = participant
    genparam['nparts'] = nparts
    genparam['ntrials'] = ntrials
    genparam['N'] = N
    sio.savemat('data/basic_ddm_dc_test1.mat', genparam)
else:
    genparam = sio.loadmat('data/basic_ddm_dc_test1.mat')

# Stan code

tostan = '''
functions { 
  /* Wiener diffusion log-PDF for a single response (adapted from brms 1.10.2)
   * Arguments: 
   *   Y: acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
   *   boundary: boundary separation parameter > 0
   *   ter: non-decision time parameter > 0
   *   bias: initial bias parameter in [0, 1]
   *   drift: drift rate parameter
   *   dc: diffusion coefficient parameter
   * Returns:  
   *   a scalar to be added to the log posterior 
   */ 
   real diffusion_lpdf(real Y, real boundary, 
                              real ter, real bias, real drift, real dc) { 
    
    if (fabs(Y) < ter) {
        return wiener_lpdf( ter+0.0001 | boundary/dc, ter, bias, drift/dc ); // does this work?
    } else {
        if (Y >= 0) {
            return wiener_lpdf( fabs(Y) | boundary/dc, ter, bias, drift/dc );
        } else {
            return wiener_lpdf( fabs(Y) | boundary/dc, ter, 1-bias, -drift/dc );
        }
    }

   }
} 
data {
    int<lower=1> N; // Number of trial-level observations
    int<lower=1> nparts; // Number of participants
    real y[N]; // acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
    int<lower=1> participant[N]; // Participant index
}
parameters {
    vector<lower=0, upper=10>[nparts] alpha; // Boundary parameter (speed-accuracy tradeoff)
    vector<lower=0, upper=1.5>[nparts] ndt; // Non-decision time
    vector<lower=0, upper=1>[nparts] beta; // Start point bias towards choice A
    vector[nparts] delta; // Drift rate to choice A
    vector<lower=0, upper=10>[nparts] varsigma; // Diffusion coefficient per participant
}
model {

    // ##########
    // Participant-level DDM parameter priors
    // ##########
    for (p in 1:nparts) {

        // Boundary parameter (speed-accuracy tradeoff) per participant
        alpha[p] ~ normal(1, .5) T[0, 10];

        // Non-decision time per participant
        ndt[p] ~ normal(.5, .25) T[0, 1.5];

        // Start point bias towards choice A per participant
        beta[p] ~ beta(2, 2);

        // Drift rate to choice A per participant
        delta[p] ~ normal(0, 2);

        // Diffusion coefficient per participant
        varsigma[p] ~ normal(1, .5) T[0, 10];

    }
    // Wiener likelihood
    for (i in 1:N) {

        target += diffusion_lpdf( y[i] | alpha[participant[i]], 
            ndt[participant[i]], beta[participant[i]], delta[participant[i]], varsigma[participant[i]]);
    }
}
'''

# pystan code

# If the model fit path does not exist, create it
model_path = f"modelfits"
if not os.path.exists(model_path):
    os.makedirs(model_path)

modelstring = 'modelfits/basic_ddm_dc_pystan2.mat'

# Set random seed
np.random.seed(2023)

nchains = 6
burnin = 2000
nsamps = 10000

# If the Stan code path does not exist, create it
stan_path = f"stancode"
if not os.path.exists(stan_path):
    os.makedirs(stan_path)

modelfile = f'stancode/basic_ddm_dc_test.stan'
f = open(modelfile, 'w')
f.write(tostan)
f.close()

# Track these variables
trackvars = ['alpha', 'ndt', 'beta', 'delta', 'varsigma']

N = np.squeeze(genparam['N'])

#Fit model to data
y = np.squeeze(genparam['y'])
rt = np.squeeze(genparam['rt'])
participant = np.array(np.squeeze(genparam['participant']),dtype=int)
nparts = np.squeeze(genparam['nparts'])
ntrials = np.squeeze(genparam['ntrials'])


#Fit model to data
data = {'y': y, 'N':N, 'nparts': nparts, 'participant': participant};

minrt = np.zeros(nparts)
for p in range(0,nparts):
    minrt[p] = np.min(rt[(participant == (p+1))])


if not os.path.exists(modelstring):
    initials = []
    for c in range(0, nchains):
        chaininit = {
            'alpha': np.random.uniform(.5, 2., size=nparts),
            'ndt': np.random.uniform(.1, .5, size=nparts),
            'beta': np.random.uniform(.2, .8, size=nparts),
            'delta': np.random.uniform(-4., 4., size=nparts),
            'varsigma': np.random.uniform(.5, 2., size=nparts),
        }
        for p in range(0, nparts):
            chaininit['ndt'][p] = np.random.uniform(0., minrt[p]/2)
        initials.append(chaininit)


    print('Fitting ''basic_ddm_dc'' model in Stan...')

    # pystan 2
    sm = pystan.StanModel(model_code=tostan)
    fit = sm.sampling(data=data, pars=trackvars, iter=nsamps+burnin, warmup=burnin, thin=10, init=initials, chains=nchains,  n_jobs=nchains, seed=2020)
    extractedsamps = fit.extract(permuted=False, pars=trackvars)
    samples = phju.flipstanout(extractedsamps)


    print('Saving results to: \n %s' % modelstring)
    sio.savemat(modelstring, samples)
else:
    print('Loading results from: \n %s' % modelstring)
    samples = sio.loadmat(modelstring)

#Diagnostics
diags = phju.diagnostic(samples)

# If the recovery plot path does not exist, create it
plot_path = f"recovery_plots/basic_ddm_dc_pystan2"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)



# Posterior distributions
plt.figure()
phju.jellyfish(samples['alpha'])
plt.title('Posterior distributions of boundary parameter')
plt.savefig(f'{plot_path}/alpha_posteriors.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.jellyfish(samples['ndt'])
plt.title('Posterior distributions of the non-decision time parameter')
plt.savefig(f'{plot_path}/ndt_posteriors.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.jellyfish(samples['beta'])
plt.title('Posterior distributions of the start point parameter')
plt.savefig(f'{plot_path}/beta_posteriors.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.jellyfish(samples['delta'])
plt.title('Posterior distributions of the drift-rate')
plt.savefig(f'{plot_path}/delta_posteriors.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.jellyfish(samples['varsigma'])
plt.title('Posterior distributions of diffusion coefficient')
plt.savefig(f'{plot_path}/varsigma_posteriors.png', format='png', bbox_inches="tight")
plt.close()

scatter_color = '#ABB0B8'

npossamps = int((nsamps/10)*nchains)

# By default plot only the first 18 random posterior draws
nplots = 18
phju.plot_posterior2d(np.reshape(samples['varsigma'][0:nplots,:,:],(nplots,npossamps)),
    np.reshape(samples['alpha'][0:nplots,:,:],(nplots,npossamps)),
   ['Diffusion coefficient', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/2d_posteriors_boundary_dc.png")
plt.close()

phju.plot_posterior2d(np.reshape(samples['delta'][0:nplots,:,:],(nplots,npossamps)),
    np.reshape(samples['alpha'][0:nplots,:,:],(nplots,npossamps)),
   ['Drift rate', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/2d_posteriors_boundary_drift.png")
plt.close()

phju.plot_posterior2d(np.reshape(samples['varsigma'][0:nplots,:,:],(nplots,npossamps)),
    np.reshape(samples['delta'][0:nplots,:,:],(nplots,npossamps)),
   ['Diffusion coefficient', 'Drift rate'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/2d_posteriors_drift_dc.png")
plt.close()

phju.plot_posterior2d(np.reshape(samples['ndt'][0:nplots,:,:],(nplots,npossamps)),
    np.reshape(samples['delta'][0:nplots,:,:],(nplots,npossamps)),
   ['Non-decision time', 'Drift rate'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/2d_posteriors_drift_ndt.png")
plt.close()

phju.plot_posterior2d(np.reshape(samples['ndt'][0:nplots,:,:],(nplots,npossamps)),
    np.reshape(samples['alpha'][0:nplots,:,:],(nplots,npossamps)),
   ['Non-decision time', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/2d_posteriors_boundary_ndt.png")
plt.close()

phju.plot_posterior2d(np.reshape(samples['beta'][0:nplots,:,:],(nplots,npossamps)),
    np.reshape(samples['alpha'][0:nplots,:,:],(nplots,npossamps)),
   ['Start point', 'Boundary'],
   font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color)
plt.savefig(f"{plot_path}/2d_posteriors_boundary_start.png")
plt.close()

# Plot a 3D joint posterior
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

main_color = '#332288'
secondary_color = '#ABB0B8'

# By default plot only one random posterior draws, draw 7
rand_draw = 17

# Main 3D scatter plot
ax.scatter(samples['delta'][rand_draw,:,:].squeeze(),
           samples['alpha'][rand_draw,:,:].squeeze(),
           samples['varsigma'][rand_draw,:,:].squeeze(), 
           alpha=0.25, color=main_color)

# 2D scatter plot for drift rate and boundary (xy plane) at min diffusion coefficient
min_dc = samples['varsigma'][rand_draw,:,:].squeeze().min()
ax.scatter(samples['delta'][rand_draw,:,:].squeeze(), 
    samples['alpha'][rand_draw,:,:].squeeze(), 
    min_dc, alpha=0.25, color=secondary_color)

# 2D scatter plot for drift rate and diffusion coefficient (xz plane) at max boundary
max_boundary = samples['alpha'][rand_draw,:,:].squeeze().max()
ax.scatter(samples['delta'][rand_draw,:,:].squeeze(), max_boundary, 
    samples['varsigma'][rand_draw,:,:].squeeze(), alpha=0.25, color=secondary_color)

# 2D scatter plot for boundary and diffusion coefficient (yz plane) at min drift rate
min_drift_rate = samples['delta'][rand_draw,:,:].squeeze().min()
ax.scatter(min_drift_rate, samples['alpha'][rand_draw,:,:].squeeze(), 
    samples['varsigma'][rand_draw,:,:].squeeze(), alpha=0.25, color=secondary_color)

ax.set_xlabel(r'Drift rate ($\delta$)', fontsize=16, labelpad=10)
ax.set_ylabel(r'Boundary ($\alpha$)', fontsize=16, labelpad=10)
ax.set_zlabel(r'Diffusion coefficient ($\varsigma$)', fontsize=16, labelpad=10)

# Rotate the plot slightly clockwise around the z-axis
elevation = 20  # Default elevation
azimuth = -30   # Rotate 30 degrees counterclockwise from the default azimuth (which is -90)
ax.view_init(elev=elevation, azim=azimuth)

plt.savefig(f"{plot_path}/3d_posterior_drift_boundary_dc.png", dpi=300,
    bbox_inches="tight", pad_inches=0.5)
plt.close()

publication_text = rf"""
Draws from a joint posterior distribution for one simulated data set from a DDM with all three 
parameters free to vary (purple 3D scatter plot). Paired joint distributions are given by the grey projections
on each of the three faces. The joint posterior distribution is driven mostly by the joint likelihood
of the data (N={int(ntrials)}) given the model (Model dcDDM). The prior distributions 
(though not influential) for Model dcDDM are given in the text. The posterior shape will be different for each data set 
(see Figure for paired posterior distributions). The true 5-dimension joint posterior distribution also includes the 
relative start point and non-decision time. The mean posteriors of those two parameters were 
$\hat\tau={(diags['ndt']['mean'][rand_draw]):.3}$ seconds and $\hat\beta={(diags['beta']['mean'][rand_draw]):.2f}$ 
proportion of boundary in this simulation respectively. The drift rate $\delta$ and diffusion coefficients $\varsigma$ are in are 
evidence units per second while the boundary $\alpha$ is in evidence units.
"""
print(publication_text)

# Recovery
plt.figure()
phju.recovery(samples['alpha'], genparam['alpha'])
plt.title('Recovery of boundary parameter')
plt.savefig(f'{plot_path}/alpha_recovery.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.recovery(samples['ndt'], genparam['ndt'])
plt.title('Recovery of the non-decision time parameter')
plt.savefig(f'{plot_path}/ndt_recovery.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.recovery(samples['beta'], genparam['beta'])
plt.title('Recovery of the start point parameter')
plt.savefig(f'{plot_path}/beta_recovery.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.recovery(samples['delta'], genparam['delta'])
plt.title('Recovery of the drift-rate')
plt.savefig(f'{plot_path}/delta_recovery.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.recovery(samples['varsigma'], genparam['varsigma'])
plt.title('Recovery of the diffusion coefficient')
plt.savefig(f'{plot_path}/varsigma_recovery.png', format='png', bbox_inches="tight")
plt.close()

# Plot true versus estimated for a subset of parameters

true_params = np.vstack((genparam['delta'],genparam['varsigma'],
    genparam['alpha'],genparam['beta'],genparam['ndt'])).T
param_means = np.vstack((diags['delta']['mean'],diags['varsigma']['mean'],
    diags['alpha']['mean'],diags['beta']['mean'],diags['ndt']['mean'])).T
phju.recovery_scatter(true_params,
                  param_means,
                  ['Drift Rate', 'Diffusion Coefficient', 'Boundary',
                  'Start Point', 'Non-Decision Time'],
                  font_size=16, color='#3182bdff', alpha=0.75, grantB1=False)
plt.savefig(f"{plot_path}/recovery_short.png")
plt.close()
