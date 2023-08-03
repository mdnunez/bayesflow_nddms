# mean_RT_accuracy_effects.py - Simulate directly from Wiener process and plot parameters versus accuracy and RT
#
# Copyright (C) 2023 Michael D. Nunez, <m.d.nunez@uva.nl>
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
# 05-July-23     Michael Nunez    Converted from nng_diffusion_coefficient1.py
# 27-July-23     Michael Nunez                            Figure for paper

# Load Modules
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Make nice subplots for the paper
rows = 3
columns = 3
fontsize = 16
f, axarr = plt.subplots(rows, columns, sharex='col', figsize=(15,10), tight_layout=True)

# Directly simulated Drift Diffusion model with the true diffusion coefficient changing
nmodelsims = 50

ntrials = 200
nsteps = 300
step_length = .01
base_boundary = 1
boundary = base_boundary
base_drift = 3
drift = base_drift
ndt = 0.2
diffusion_coefficients = np.linspace(0.6,1.4,nmodelsims)
mean_accuracy = np.empty((nmodelsims))
mean_rt = np.empty((nmodelsims))
var_rt = np.empty((nmodelsims))

np.random.seed(1)  # Try different random seeds

for sim in range(nmodelsims):
    dc = diffusion_coefficients[sim]
    rts = np.empty(ntrials)  # Simulated correct response times
    correct = np.empty(ntrials)
    # if sim == 1:
    #     plt.figure()
    time = np.linspace(0, step_length*nsteps, num=nsteps)
    for n in range(ntrials):
        random_walk = np.empty(nsteps)
        random_walk[0] = 0.5*boundary  # relative start point = 0.5
        for s in range(1,nsteps):
            random_walk[s] = random_walk[s-1] + stats.norm.rvs(loc=drift*step_length, scale=dc*np.sqrt(step_length))
            if random_walk[s] >= boundary:
                random_walk[s:] = boundary
                rts[n] = s*step_length + ndt
                correct[n] = 1
                break
            elif random_walk[s] <= 0:
                random_walk[s:] = 0
                rts[n] = s*step_length + ndt
                correct[n] = 0
                break
            elif s == (nsteps-1):
                correct[n] = np.nan; rts[n] = np.nan
                break
    #     if sim == 1:
    #         plt.plot(time + ndt, random_walk)
    # if sim == 1:
    #     plt.xlim([0, 2]); plt.xlabel('Time (secs)'); plt.ylabel('Cognitive / Neural Evidence')
    #     plt.figure()
    #     plt.hist((correct * 2 - 1) * rts, bins=20)  # Typical method of plotting choice-RTs
    #     plt.xlabel('Choice * Response Time (sec)')
    # print(np.nanmean(correct))
    # print(np.nanmean(rts))
    mean_accuracy[sim] = np.nanmean(correct)
    mean_rt[sim] = np.nanmean(rts)
    var_rt[sim] = np.nanvar(rts)



# Make subplots
axarr[0,1].scatter(diffusion_coefficients, mean_accuracy)
#axarr[0,1].set_ylabel('Mean accuracy', fontsize=fontsize)
#axarr[0,1].set_xlabel('Diffusion Coefficient', fontsize=fontsize)
axarr[0,1].tick_params(axis='both', labelsize=fontsize)

axarr[1,1].scatter(diffusion_coefficients, mean_rt)
#axarr[1,1].set_ylabel('Mean RT (secs)', fontsize=fontsize)
#axarr[1,1].set_xlabel('Diffusion Coefficient', fontsize=fontsize)
axarr[1,1].tick_params(axis='both', labelsize=fontsize)

axarr[2,1].scatter(diffusion_coefficients, var_rt)
#axarr[2,1].set_ylabel('RT variance (secs$^2$)', fontsize=fontsize)
axarr[2,1].set_xlabel('Diffusion Coefficient', fontsize=fontsize)
axarr[2,1].tick_params(axis='both', labelsize=fontsize)

# # Directly simulated Drift Diffusion model with the scalar on boundary and drift changing
# mean_accuracy = np.empty((nmodelsims))
# mean_rt = np.empty((nmodelsims))
# var_rt = np.empty((nmodelsims))

# np.random.seed(1)  # Try different random seeds

# for sim in range(nmodelsims):
#     scalar = diffusion_coefficients[sim]
#     dc = 1
#     boundary = base_boundary/scalar
#     drift = base_drift/scalar
#     rts = np.empty(ntrials)  # Simulated correct response times
#     correct = np.empty(ntrials)
#     if sim == 1:
#         plt.figure()
#     time = np.linspace(0, step_length*nsteps, num=nsteps)
#     for n in range(ntrials):
#         random_walk = np.empty(nsteps)
#         random_walk[0] = 0.5*boundary  # relative start point = 0.5
#         for s in range(1,nsteps):
#             random_walk[s] = random_walk[s-1] + stats.norm.rvs(loc=drift*step_length, scale=dc*np.sqrt(step_length))
#             if random_walk[s] >= boundary:
#                 random_walk[s:] = boundary
#                 rts[n] = s*step_length + ndt
#                 correct[n] = 1
#                 break
#             elif random_walk[s] <= 0:
#                 random_walk[s:] = 0
#                 rts[n] = s*step_length + ndt
#                 correct[n] = 0
#                 break
#             elif s == (nsteps-1):
#                 correct[n] = np.nan; rts[n] = np.nan
#                 break
#         if sim == 1:
#             plt.plot(time + ndt, random_walk)
#     if sim == 1:
#         plt.xlim([0, 2]); plt.xlabel('Time (secs)'); plt.ylabel('Cognitive / Neural Evidence')
#         plt.figure()
#         plt.hist((correct * 2 - 1) * rts, bins=20)  # Typical method of plotting choice-RTs
#         plt.xlabel('Choice * Response Time (sec)')
#     print(np.nanmean(correct))
#     print(np.nanmean(rts))
#     mean_accuracy[sim] = np.nanmean(correct)
#     mean_rt[sim] = np.nanmean(rts)
#     var_rt[sim] = np.nanvar(rts)



# #Plot diffusion coefficient versus accuracy and response time
# ax4.scatter(diffusion_coefficients, mean_accuracy)
# ax4.set_ylabel('Mean accuracy', fontsize=fontsize)
# ax4.set_xlabel('Scalar on drift and boundary', fontsize=fontsize)

# ax5.scatter(diffusion_coefficients, mean_rt)
# ax5.set_ylabel('Mean RT (secs)', fontsize=fontsize)
# ax5.set_xlabel('Scalar on drift and boundary', fontsize=fontsize)

# ax6.scatter(diffusion_coefficients, var_rt)
# ax6.set_ylabel('RT variance (secs$^2$)', fontsize=fontsize)
# ax6.set_xlabel('Scalar on drift and boundary', fontsize=fontsize)



# Directly simulated Drift Diffusion model with the scalar only on drift rate changing
mean_accuracy = np.empty((nmodelsims))
mean_rt = np.empty((nmodelsims))
var_rt = np.empty((nmodelsims))

np.random.seed(1)  # Try different random seeds

for sim in range(nmodelsims):
    scalar = diffusion_coefficients[sim]
    dc = 1
    boundary = base_boundary
    drift = base_drift/scalar
    rts = np.empty(ntrials)  # Simulated correct response times
    correct = np.empty(ntrials)
    # if sim == 1:
    #     plt.figure()
    time = np.linspace(0, step_length*nsteps, num=nsteps)
    for n in range(ntrials):
        random_walk = np.empty(nsteps)
        random_walk[0] = 0.5*boundary  # relative start point = 0.5
        for s in range(1,nsteps):
            random_walk[s] = random_walk[s-1] + stats.norm.rvs(loc=drift*step_length, scale=dc*np.sqrt(step_length))
            if random_walk[s] >= boundary:
                random_walk[s:] = boundary
                rts[n] = s*step_length + ndt
                correct[n] = 1
                break
            elif random_walk[s] <= 0:
                random_walk[s:] = 0
                rts[n] = s*step_length + ndt
                correct[n] = 0
                break
            elif s == (nsteps-1):
                correct[n] = np.nan; rts[n] = np.nan
                break
    #     if sim == 1:
    #         plt.plot(time + ndt, random_walk)
    # if sim == 1:
    #     plt.xlim([0, 2]); plt.xlabel('Time (secs)'); plt.ylabel('Cognitive / Neural Evidence')
    #     plt.figure()
    #     plt.hist((correct * 2 - 1) * rts, bins=20)  # Typical method of plotting choice-RTs
    #     plt.xlabel('Choice * Response Time (sec)')
    # print(np.nanmean(correct))
    # print(np.nanmean(rts))
    mean_accuracy[sim] = np.nanmean(correct)
    mean_rt[sim] = np.nanmean(rts)
    var_rt[sim] = np.nanvar(rts)



#Plot scalar on drift versus accuracy and response time
axarr[0,0].scatter(base_drift/diffusion_coefficients, mean_accuracy)
axarr[0,0].set_ylabel('Mean accuracy', fontsize=fontsize)
#axarr[0,0].set_xlabel('Drift Rate', fontsize=fontsize)
axarr[0,0].tick_params(axis='both', labelsize=fontsize)

axarr[1,0].scatter(base_drift/diffusion_coefficients, mean_rt)
axarr[1,0].set_ylabel('Mean RT (secs)', fontsize=fontsize)
#axarr[1,0].set_xlabel('Drift Rate', fontsize=fontsize)
axarr[1,0].tick_params(axis='both', labelsize=fontsize)

axarr[2,0].scatter(base_drift/diffusion_coefficients, var_rt)
axarr[2,0].set_ylabel('RT variance (secs$^2$)', fontsize=fontsize)
axarr[2,0].set_xlabel('Drift Rate', fontsize=fontsize)
axarr[2,0].tick_params(axis='both', labelsize=fontsize)

# Directly simulated Drift Diffusion model with the scalar only on boundary changing
mean_accuracy = np.empty((nmodelsims))
mean_rt = np.empty((nmodelsims))
var_rt = np.empty((nmodelsims))


np.random.seed(1)  # Try different random seeds

for sim in range(nmodelsims):
    scalar = diffusion_coefficients[sim]
    dc = 1
    boundary = base_boundary/scalar
    drift = base_drift
    rts = np.empty(ntrials)  # Simulated correct response times
    correct = np.empty(ntrials)
    # if sim == 1:
    #     plt.figure()
    time = np.linspace(0, step_length*nsteps, num=nsteps)
    for n in range(ntrials):
        random_walk = np.empty(nsteps)
        random_walk[0] = 0.5*boundary  # relative start point = 0.5
        for s in range(1,nsteps):
            random_walk[s] = random_walk[s-1] + stats.norm.rvs(loc=drift*step_length, scale=dc*np.sqrt(step_length))
            if random_walk[s] >= boundary:
                random_walk[s:] = boundary
                rts[n] = s*step_length + ndt
                correct[n] = 1
                break
            elif random_walk[s] <= 0:
                random_walk[s:] = 0
                rts[n] = s*step_length + ndt
                correct[n] = 0
                break
            elif s == (nsteps-1):
                correct[n] = np.nan; rts[n] = np.nan
                break
    #     if sim == 1:
    #         plt.plot(time + ndt, random_walk)
    # if sim == 1:
    #     plt.xlim([0, 2]); plt.xlabel('Time (secs)'); plt.ylabel('Cognitive / Neural Evidence')
    #     plt.figure()
    #     plt.hist((correct * 2 - 1) * rts, bins=20)  # Typical method of plotting choice-RTs
    #     plt.xlabel('Choice * Response Time (sec)')
    # print(np.nanmean(correct))
    # print(np.nanmean(rts))
    mean_accuracy[sim] = np.nanmean(correct)
    mean_rt[sim] = np.nanmean(rts)
    var_rt[sim] = np.nanvar(rts)



#Plot scalar on boundary versus accuracy and response time
axarr[0,2].scatter(base_boundary/diffusion_coefficients, mean_accuracy)
#axarr[0,2].set_ylabel('Mean accuracy', fontsize=fontsize)
#axarr[0,2].set_xlabel('Boundary', fontsize=fontsize)
axarr[0,2].tick_params(axis='both', labelsize=fontsize)

axarr[1,2].scatter(base_boundary/diffusion_coefficients, mean_rt)
#axarr[1,2].set_ylabel('Mean RT (secs)', fontsize=fontsize)
#axarr[1,2].set_xlabel('Boundary', fontsize=fontsize)
axarr[1,2].tick_params(axis='both', labelsize=fontsize)

axarr[2,2].scatter(base_boundary/diffusion_coefficients, var_rt)
#axarr[2,2].set_ylabel('RT variance (secs$^2$)', fontsize=fontsize)
axarr[2,2].set_xlabel('Boundary', fontsize=fontsize)
axarr[2,2].tick_params(axis='both', labelsize=fontsize)


# Tight layout with no space between plots
f.tight_layout(h_pad=0, w_pad=0)

plt.savefig("mean_RT_accuracy_effects.png", dpi=300)
