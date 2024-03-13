# alpha_not_scaled.py - Testing JAGS fits of a non-hierarchical DDM model with
#                              external data predicted by boundary in Python 3
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
# 06/10/23      Michael Nunez                            Original code
# 09/10/23      Michael Nunez         Fixed sigma to be standard deviation
# 10/10/23      Michael Nunez  Do not output all posterior distribution plots
# 11/03/24      Michael Nunez  Updated plots from pyhddmjagsutils, fix draw across tests
# 13/03/24      Michael Nunez    Create empty directories if they do not exist

# MODULES
import numpy as np
import pyjags
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import pyhddmjagsutils as phju

# FLAGS

test_num = 2
print(f'Obtaining model fits for test {test_num}...')

# SIMULATE MODEL

# If the simulation data path does not exist, create it
data_path = f"data"
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Generate samples from the joint-model of reaction time and choice
# Note you could remove this if statement and replace with loading your own data to dictionary "gendata"

if not os.path.exists(f'data/alpha_not_scaled_test{test_num}.mat'):
    print(f'Generated simulated data for test {test_num}...')
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
    var_alpha = (1/12)*(1.4 - .8)**2 #0.03
    beta = np.random.uniform(.3, .7, size=nparts)  # Uniform from .3 to .7 * alpha
    delta = np.random.uniform(-4, 4, size=nparts)  # Uniform from -4 to 4 evidence units per second
    varsigma = np.random.uniform(.8, 1.4, size=nparts) # Uniform from .8 to 1.4 evidence units
    deltatrialsd = np.random.uniform(0, 2, size=nparts)  # Uniform from 0 to 2 evidence units per second
    if (test_num == 1):
        sigma = .5 # Test 1, high measurement noise compared to standard deviation of boundaries
    elif (test_num == 2):
        sigma = .1 # Test 2, low measurement noise compared to standard deviation of boundaries
    elif (test_num == 3):
        sigma = .01 # Test 3, very low measurement noise compared to standard deviation of boundaries
    elif (test_num == 4):
        sigma = .2 # Test 4, simulate no connection to external covariate
        # Note this exactly matches the total variance of test 2: sqrt(0.03 + 0.01)
    # # Fix parameters across simulations
    ndt[17] = .4
    alpha[17] = 1.2
    beta[17] = .5
    delta[17] = 3.5
    varsigma[17] = 1.2
    deltatrialsd[17] = 1
    y = np.zeros(N)
    rt = np.zeros(N)
    acc = np.zeros(N)
    extdata = np.zeros(nparts)  # External data measured per participant
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
        if test_num != 4:
            extdata[p] = np.random.normal(loc=1*alpha[p], scale=sigma)
        else:
            extdata[p] = np.random.normal(loc=1, scale=sigma)
        participant[indextrack] = p + 1
        indextrack += ntrials

    genparam = dict()
    genparam['ndt'] = ndt
    genparam['beta'] = beta
    genparam['alpha'] = alpha
    genparam['delta'] = delta
    genparam['deltatrialsd'] = deltatrialsd
    genparam['varsigma'] = varsigma
    genparam['sigma'] = sigma
    genparam['var_alpha'] = var_alpha
    genparam['prop_cog_var'] = var_alpha / (var_alpha + sigma**2)
    genparam['rt'] = rt
    genparam['acc'] = acc
    genparam['y'] = y
    genparam['extdata'] = extdata
    genparam['participant'] = participant
    genparam['nparts'] = nparts
    genparam['ntrials'] = ntrials
    genparam['N'] = N
    sio.savemat(f'data/alpha_not_scaled_test{test_num}.mat', genparam)
else:
    print(f'Loading pre-simulated data for test {test_num}...')
    genparam = sio.loadmat(f'data/alpha_not_scaled_test{test_num}.mat')

# JAGS code

# Set random seed
np.random.seed(2020)

tojags = '''
model {
    
    ##########
    sigma ~ dnorm(3,pow(1, -2))T(0, 10)
    
    ##########
    #Simple DDM parameter priors
    ##########
    for (p in 1:nparts) {
    
        #Boundary parameter (speed-accuracy tradeoff) per participant
        alpha[p] ~ dnorm(1, pow(.5,-2))T(0, 10)

        #Non-decision time per participant
        ndt[p] ~ dnorm(.5, pow(.25,-2))T(0, 1.5)

        #Start point bias towards choice A per participant
        beta[p] ~ dbeta(2, 2)

        #Drift rate to choice A per participant
        delta[p] ~ dnorm(0, pow(2, -2))

        #Diffusion coefficient per participant
        varsigma[p] ~ dnorm(1, pow(.5,-2))T(0, 10)

        # Obervations of external data per participant
        extdata[p] ~ dnorm(1*alpha[p], pow(sigma,-2))

    }

    ##########
    # Wiener likelihood
    for (i in 1:N) {

        # Observations of accuracy*RT for DDM process of rightward/leftward RT
        y[i] ~ dwiener(alpha[participant[i]]/varsigma[participant[i]], 
        ndt[participant[i]], beta[participant[i]], 
        delta[participant[i]]/varsigma[participant[i]])
        

    }
}
'''

# pyjags code

# If the model fit path does not exist, create it
model_path = f"modelfits"
if not os.path.exists(model_path):
    os.makedirs(model_path)

modelstring = f'modelfits/alpha_not_scaled_test{test_num}.mat'


# Make sure $LD_LIBRARY_PATH sees /usr/local/lib
# Make sure that the correct JAGS/modules-4/ folder contains wiener.so and wiener.la
pyjags.modules.load_module('wiener')
pyjags.modules.load_module('dic')
pyjags.modules.list_modules()

nchains = 6
burnin = 2000  # Note that scientific notation breaks pyjags
nsamps = 10000

# If the JAGS code path does not exist, create it
jags_path = f"jagscode"
if not os.path.exists(jags_path):
    os.makedirs(jags_path)

modelfile = f'jagscode/alpha_not_scaled_test{test_num}.jags'
f = open(modelfile, 'w')
f.write(tojags)
f.close()

# Track these variables
trackvars = ['alpha', 'ndt', 'beta', 'delta', 'varsigma', 'sigma']

N = np.squeeze(genparam['N'])

# Fit model to data
y = np.squeeze(genparam['y'])
rt = np.squeeze(genparam['rt'])
participant = np.squeeze(genparam['participant'])
nparts = np.squeeze(genparam['nparts'])
ntrials = np.squeeze(genparam['ntrials'])
extdata = np.squeeze(genparam['extdata'])

minrt = np.zeros(nparts)
for p in range(0, nparts):
    minrt[p] = np.min(rt[(participant == (p + 1))])

if not os.path.exists(modelstring):
    print(f'Fitting {modelstring}...')
    initials = []
    for c in range(0, nchains):
        chaininit = {
            'alpha': np.random.uniform(.5, 2., size=nparts),
            'ndt': np.random.uniform(.1, .5, size=nparts),
            'beta': np.random.uniform(.2, .8, size=nparts),
            'delta': np.random.uniform(-4., 4., size=nparts),
            'varsigma': np.random.uniform(.5, 2., size=nparts),
            'sigma': np.random.uniform(0.01, 5., size=1)
        }
        for p in range(0, nparts):
            chaininit['ndt'][p] = np.random.uniform(0., minrt[p] / 2)
        initials.append(chaininit)
    print('Fitting ''alpha_not_scaled'' model ...')
    threaded = pyjags.Model(file=modelfile, init=initials,
                            data=dict(y=y, N=N, extdata=extdata, nparts=nparts,
                                      participant=participant),
                            chains=nchains, adapt=burnin, threads=6,
                            progress_bar=True)
    samples = threaded.sample(nsamps, vars=trackvars, thin=10)
    print('Saving results to: \n %s' % modelstring)
    sio.savemat(modelstring, samples)
else:
    print('Loading results from: \n %s' % modelstring)
    samples = sio.loadmat(modelstring)

# Diagnostics
diags = phju.diagnostic(samples)


# MAKE RECOVERY PLOTS

model_name = 'alpha_not_scaled'

# If the recovery plot path does not exist, create it
plot_path = f"recovery_plots/alpha_not_scaled_test{test_num}"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)


true_params = np.vstack((genparam['delta'],genparam['varsigma'],
    genparam['alpha'],genparam['beta'],genparam['ndt'])).T
param_means = np.vstack((diags['delta']['mean'],diags['varsigma']['mean'],
    diags['alpha']['mean'],diags['beta']['mean'],diags['ndt']['mean'])).T

# Posterior distributions
plt.figure()
phju.jellyfish(samples['alpha'])
plt.title('Posterior distributions of boundary parameter')
plt.savefig(f'{plot_path}/{model_name}_alpha_posteriors.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.jellyfish(samples['ndt'])
plt.title('Posterior distributions of the non-decision time parameter')
plt.savefig(f'{plot_path}/{model_name}_ndt_posteriors.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.jellyfish(samples['beta'])
plt.title('Posterior distributions of the start point parameter')
plt.savefig(f'{plot_path}/{model_name}_beta_posteriors.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.jellyfish(samples['delta'])
plt.title('Posterior distributions of the drift-rate')
plt.savefig(f'{plot_path}/{model_name}_delta_posteriors.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.jellyfish(samples['varsigma'])
plt.title('Posterior distributions of diffusion coefficient')
plt.savefig(f'{plot_path}/{model_name}_varsigma_posteriors.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.jellyfish(samples['sigma'])
plt.title('Posterior distribution of measurement noise in external data')
plt.savefig(f'{plot_path}/{model_name}_sigma_posterior.png', format='png', bbox_inches="tight")
plt.close()

scatter_color = '#ABB0B8'

npossamps = int((nsamps/10)*nchains)

# By default plot only one random posterior draws, draw 7
rand_draw = 17

# By default plot only the first 18 random posterior draws
nplots = 18
phju.plot_posterior2d(np.reshape(samples['varsigma'][rand_draw:(nplots+rand_draw),:,:],(nplots,npossamps)),
    np.reshape(samples['alpha'][rand_draw:(nplots+rand_draw),:,:],(nplots,npossamps)),
   ['Diffusion coefficient', 'Boundary'],
       true_params=true_params[:, np.array([1, 2])][rand_draw:(nplots+rand_draw), :],
       font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color,
       color2='black', highlight=rand_draw)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_dc.png")
plt.close()

phju.plot_posterior2d(np.reshape(samples['delta'][rand_draw:(nplots+rand_draw),:,:],(nplots,npossamps)),
    np.reshape(samples['alpha'][rand_draw:(nplots+rand_draw),:,:],(nplots,npossamps)),
   ['Drift rate', 'Boundary'],
       true_params=true_params[:, np.array([0, 2])][rand_draw:(nplots+rand_draw), :],
       font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color,
       color2='black', highlight=rand_draw)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_drift.png")
plt.close()

phju.plot_posterior2d(np.reshape(samples['varsigma'][rand_draw:(nplots+rand_draw),:,:],(nplots,npossamps)),
    np.reshape(samples['delta'][rand_draw:(nplots+rand_draw),:,:],(nplots,npossamps)),
   ['Diffusion coefficient', 'Drift rate'],
       true_params=true_params[:, np.array([1, 0])][rand_draw:(nplots+rand_draw), :],
       font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color,
       color2='black', highlight=rand_draw)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_drift_dc.png")
plt.close()

phju.plot_posterior2d(np.reshape(samples['ndt'][rand_draw:(nplots+rand_draw),:,:],(nplots,npossamps)),
    np.reshape(samples['delta'][rand_draw:(nplots+rand_draw),:,:],(nplots,npossamps)),
   ['Non-decision time', 'Drift rate'],
       true_params=true_params[:, np.array([4, 0])][rand_draw:(nplots+rand_draw), :],
       font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color,
       color2='black', highlight=rand_draw)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_drift_ndt.png")
plt.close()

phju.plot_posterior2d(np.reshape(samples['ndt'][rand_draw:(nplots+rand_draw),:,:],(nplots,npossamps)),
    np.reshape(samples['alpha'][rand_draw:(nplots+rand_draw),:,:],(nplots,npossamps)),
   ['Non-decision time', 'Boundary'],
       true_params=true_params[:, np.array([4, 2])][rand_draw:(nplots+rand_draw), :],
       font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color,
       color2='black', highlight=rand_draw)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_ndt.png")
plt.close()

phju.plot_posterior2d(np.reshape(samples['beta'][rand_draw:(nplots+rand_draw),:,:],(nplots,npossamps)),
    np.reshape(samples['alpha'][rand_draw:(nplots+rand_draw),:,:],(nplots,npossamps)),
   ['Start point', 'Boundary'],
       true_params=true_params[:, np.array([3, 2])][rand_draw:(nplots+rand_draw), :],
       font_size=16, alpha=0.25, figsize=(20,8), color=scatter_color,
       color2='black', highlight=rand_draw)
plt.savefig(f"{plot_path}/{model_name}_2d_posteriors_boundary_start.png")
plt.close()

# Plot a 3D joint posterior
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

main_color = '#332288'
secondary_color = '#ABB0B8'



# Main 3D scatter plot
ax.scatter(samples['delta'][rand_draw,:,:].squeeze(),
           samples['alpha'][rand_draw,:,:].squeeze(),
           samples['varsigma'][rand_draw,:,:].squeeze(), 
           alpha=0.25, color=main_color)

# 2D scatter plot for drift rate and boundary (xy plane) at min diffusion coefficient
#min_dc = samples['varsigma'][rand_draw,:,:].squeeze().min()
min_dc = 0
ax.scatter(samples['delta'][rand_draw,:,:].squeeze(), 
    samples['alpha'][rand_draw,:,:].squeeze(), 
    min_dc, alpha=0.25, color=secondary_color)

# 2D scatter plot for drift rate and diffusion coefficient (xz plane) at max boundary
#max_boundary = samples['alpha'][rand_draw,:,:].squeeze().max()
max_boundary = 2.5
ax.scatter(samples['delta'][rand_draw,:,:].squeeze(), max_boundary, 
    samples['varsigma'][rand_draw,:,:].squeeze(), alpha=0.25, color=secondary_color)

# 2D scatter plot for boundary and diffusion coefficient (yz plane) at min drift rate
#min_drift_rate = samples['delta'][rand_draw,:,:].squeeze().min()
min_drift_rate = 0
ax.scatter(min_drift_rate, samples['alpha'][rand_draw,:,:].squeeze(), 
    samples['varsigma'][rand_draw,:,:].squeeze(), alpha=0.25, color=secondary_color)

ax.set_xlabel(r'Drift rate ($\delta$)', fontsize=16, labelpad=10)
ax.set_ylabel(r'Boundary ($\alpha$)', fontsize=16, labelpad=10)
ax.set_zlabel(r'Diffusion coefficient ($\varsigma$)', fontsize=16, labelpad=10)

# Rotate the plot slightly clockwise around the z-axis
elevation = 20  # Default elevation
azimuth = -30   # Rotate 30 degrees counterclockwise from the default azimuth (which is -90)
ax.view_init(elev=elevation, azim=azimuth)

# Set plot axes to compare to test 4 with unidentified posterior
ax.set_xlim([min_drift_rate, 6.5]) # Limits for drift rate
ax.set_ylim([0.4, max_boundary]) # Limits for boundary
ax.set_zlim([min_dc, 2.5]) # Limits for diffusion coefficient

plt.savefig(f"{plot_path}/{model_name}_3d_posterior_drift_boundary_dc.png", dpi=300,
    bbox_inches="tight", pad_inches=0.5)
plt.close()

sigma = np.squeeze(genparam['sigma'])
publication_text = rf"""
Draws from a joint posterior distribution for one simulated data set from a DDM with all three 
parameters free to vary (purple 3D scatter plot) with external data described by the boundary. Paired joint distributions are given by the grey projections
on each of the three faces. The joint posterior distribution is driven mostly by the joint likelihood
of the data (N={int(ntrials)}) given the model (Model dcDDM-$\alpha$) with measurement noise of external data $\sigma={(sigma):.2}$. The prior distributions 
(though not influential) for Model dcDDM-$\alpha$ are given in the text. The posterior shape will be different for each data set 
(see Figure for paired posterior distributions). The true 5-dimension joint posterior distribution also includes the 
relative start point and non-decision time. The mean posteriors of those two parameters were 
$\hat\tau={(diags['ndt']['mean'][rand_draw]):.3}$ seconds and $\hat\beta={(diags['beta']['mean'][rand_draw]):.2f}$ 
proportion of boundary in this simulation respectively. The drift rate $\delta$ and diffusion coefficients $\varsigma$ are in are 
evidence units per second while the boundary $\alpha$ is in evidence units. The true parameters were 
$\delta={(true_params[rand_draw,0]):.2f}$, $\varsigma={(true_params[rand_draw,1]):.2f}$, $\alpha={(true_params[rand_draw,2]):.2f}$, 
$\beta={(true_params[rand_draw,3]):.2f}$, and $\tau={(true_params[rand_draw,4]):.3}$.
"""
print(publication_text)

# Recovery
plt.figure()
phju.recovery(samples['alpha'], genparam['alpha'])
plt.title('Recovery of boundary parameter')
plt.savefig(f'{plot_path}/{model_name}_alpha_recovery.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.recovery(samples['ndt'], genparam['ndt'])
plt.title('Recovery of the non-decision time parameter')
plt.savefig(f'{plot_path}/{model_name}_ndt_recovery.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.recovery(samples['beta'], genparam['beta'])
plt.title('Recovery of the start point parameter')
plt.savefig(f'{plot_path}/{model_name}_beta_recovery.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.recovery(samples['delta'], genparam['delta'])
plt.title('Recovery of the drift-rate')
plt.savefig(f'{plot_path}/{model_name}_delta_recovery.png', format='png', bbox_inches="tight")
plt.close()

plt.figure()
phju.recovery(samples['varsigma'], genparam['varsigma'])
plt.title('Recovery of the diffusion coefficient')
plt.savefig(f'{plot_path}/{model_name}_varsigma_recovery.png', format='png', bbox_inches="tight")
plt.close()

# Plot true versus estimated for a subset of parameters


phju.recovery_scatter(true_params,
                  param_means,
                  ['Drift Rate', 'Diffusion Coefficient', 'Boundary',
                  'Start Point', 'Non-Decision Time'],
                  font_size=16, color='#3182bdff', alpha=0.75, grantB1=False)
plt.savefig(f"{plot_path}/{model_name}_recovery_short.png")
plt.close()


