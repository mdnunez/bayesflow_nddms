# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 12-May-2023   Michael Nunez   Converted from course code Behavioural Research Toolbox 26-Sep-22
# 15-May-2023   Michael Nunez            Additions and subplots
# 26-June-2023  Michael Nunez            Flag to plot the last row

extra_row = False


# Load Modules
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Straight-forward DDM simulation
def basic_diffusion(ntrials=200, nsteps=300, step_length=.01,
    boundary=1.2, drift=1.5, ndt=0.35, dc=1, fontsize = 16):
    """Simulates a multiple trials from a diffusion model and plots the evidence path."""

    rts = np.empty(ntrials)  # Simulated correct response times
    correct = np.empty(ntrials)
    plt.figure()
    time = np.linspace(0, step_length*nsteps, num=nsteps)
    random_walk = np.empty((nsteps, ntrials))
    plottime = time + ndt
    for n in range(ntrials):
        random_walk[0, n] = 0.5*boundary  # relative start point = 0.5
        for s in range(1,nsteps):
            random_walk[s, n] = random_walk[s-1, n] + stats.norm.rvs(loc=drift*step_length, 
                scale=dc*np.sqrt(step_length))
            if random_walk[s, n] >= boundary:
                random_walk[s:, n] = boundary
                rts[n] = s*step_length + ndt
                correct[n] = 1
                break
            elif random_walk[s, n] <= 0:
                random_walk[s:, n] = 0
                rts[n] = s*step_length + ndt
                correct[n] = 0
                break
            elif s == (nsteps-1):
                correct[n] = np.nan; rts[n] = np.nan
                break
        plt.plot(plottime, random_walk[:, n])
    plt.xlim([0, 2])
    plt.xlabel('Time (secs)', fontsize=fontsize)
    plt.ylabel('Cognitive / Neural Evidence', fontsize=fontsize)

    plt.figure()
    plt.hist((correct * 2 - 1) * rts, bins=20)  # Typical method of plotting choice-RTs
    plt.xlabel('Choice * Response Time (sec)', fontsize=fontsize)

    print(f'Accuracy: {np.nanmean(correct)}')
    print(f'Mean RT: {np.nanmean(rts)}')

    print(f'Signal to Noise Ratio: {drift/dc}')
    print(f'Criterion to Noise Ratio: {boundary/dc}')

    plottime = time + ndt

    return correct, rts, plottime, random_walk


# Fitting DDMs to data (using EZ Diffusion)
# Code by Russ Poldrack at Stanford
# https://github.com/poldrack/ezdiff/blob/master/ezdiff.py
def ezdiff(rt,correct,s=1.0):
    logit = lambda p:np.log(p/(1-p))
    assert len(rt)>0
    assert len(rt)==len(correct)
    assert np.nanmax(correct)<=1
    assert np.nanmin(correct)>=0
    pc=np.nanmean(correct)
    assert pc > 0
    # subtract or add 1/2 an error to prevent division by zero
    if pc==1.0:
        pc=1 - 1/(2*len(correct))
    if pc==0.5:
        pc=0.5 + 1/(2*len(correct))
    MRT=np.nanmean(rt[correct==1])
    VRT=np.nanvar(rt[correct==1])
    assert VRT > 0
    r=(logit(pc)*(((pc**2) * logit(pc)) - pc*logit(pc) + pc - 0.5))/VRT
    drift=np.sign(pc-0.5)*s*(r)**0.25
    boundary=(s**2 * logit(pc))/drift
    y=(-1*drift*boundary)/(s**2)
    MDT=(boundary/(2*drift))*((1-np.exp(y))/(1+np.exp(y)))
    ndt=MRT-MDT
    print(f'EZ Drift rate estimate: {drift}')
    print(f'EZ Boundary estimate: {boundary}')
    print(f'EZ NDT estimate: {ndt}')
    return([drift,boundary,ndt])


# Make plots for diffusion with standard noise assumption
np.random.seed(2023)
(correct1, rts1, plottime1, random_walk1) = basic_diffusion(boundary = 1.2, drift = 1.5,dc = 1)
# (drift/dc) = 1.5, (alpha/dc) = 1.2
ezdiff(rts1, correct1)

# This parameter set makes the exact same choice-RT and relative evidence accumulation path predictions
# But Different evidence scale!
np.random.seed(2023)
(correct2, rts2, plottime2, random_walk2) = basic_diffusion(boundary = 2.4, drift = 3,dc = 2)
# (drift/dc) = 1.5, (alpha/dc) = 1.2
ezdiff(rts2, correct2)

if extra_row:
	# This parameter set makes the exact same choice-RT and relative evidence accumulation path predictions
	# But Different evidence scale!
	np.random.seed(2023)
	(correct3, rts3, plottime3, random_walk3) = basic_diffusion(boundary = 1, drift = 1.25,dc = .83333333)
	# (drift/dc) = 1.5, (alpha/dc) = 1.2
	ezdiff(rts3, correct3)

# # This parameter set makes the exact same choice-RT and relative evidence accumulation path predictions
# # But Different evidence scale!
# np.random.seed(2023)
# (correct4, rts4, plottime4, random_walk4) = basic_diffusion(boundary = .8, drift = 1,dc = .66666666)
# # (drift/dc) = 1.5, (alpha/dc) = 1.2
# ezdiff(rts4, correct4)

# This parameter set shows that the speed-accuracy tradeoff can occur with a manipulation on both
#true drift and true diffusion coefficient, this simulation results in faster and less correct responses
np.random.seed(2023)
(correct5, rts5, plottime5, random_walk5) = basic_diffusion(boundary = 1.2, drift = 3,dc = 2)
# (drift/dc) = 1.5, (alpha/dc) = 0.6
ezdiff(rts5, correct5)

if extra_row:
	# This parameter set shows that the speed-accuracy tradeoff can occur with a manipulation on both
	#true drift and true diffusion coefficient, this simulation results in slower and more correct responses
	np.random.seed(2023)
	(correct6, rts6, plottime6, random_walk6) = basic_diffusion(boundary = 1.2, drift = 0.75,dc = 0.5)
	# (drift/dc) = 1.5, (alpha/dc) = 2.4
	ezdiff(rts6, correct6)

# However true effects on diffusion coefficient can be masked as two effects on the other parameters
np.random.seed(2023)
(correct7, rts7, plottime7, random_walk7) = basic_diffusion(boundary = 1.2, drift = 1.5,dc = 0.5)
# (drift/dc) = 3, (alpha/dc) = 2.4
ezdiff(rts7, correct7)


# What does all this mean? This means that while the boundary and drift-rate 
# may be good psychological constructs, they mean little for the actual computation in the brain.
# They also are not useful in understanding individual and conditional differences in noise.

# Make nice subplots for the paper
if extra_row:
	rows = 3
else:
	rows = 2

columns = 2
fontsize = 16
f, axarr = plt.subplots(rows, columns, sharex = True, figsize=(15,10), tight_layout=True)

# First plot
axarr[0,0].plot(plottime1, random_walk1)
axarr[0,0].set_xlim([0, 2])
axarr[0,0].set_ylabel('Evidence', fontsize=fontsize, labelpad=-20)
axarr[0,0].set_yticks(np.array([0, 1.2]))
axarr[0,0].text(0.01, 0.8, '$\\delta$=1.5',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[0,0].transAxes, 
             size=fontsize)
axarr[0,0].text(0.01, 0.675, '$\\alpha$=1.2',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[0,0].transAxes, 
             size=fontsize)
axarr[0,0].text(0.01, 0.55, '$\\varsigma$=1.0',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[0,0].transAxes, 
             size=fontsize)
axarr[0,0].text(0.01, 0.325, '$\\delta / \\varsigma $=1.5',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[0,0].transAxes, 
             size=fontsize)
axarr[0,0].text(0.01, 0.2, '$\\alpha / \\varsigma $=1.2',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[0,0].transAxes, 
             size=fontsize)
axarr[0,0].text(-0.12, 0.97, 'a', weight='bold',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[0,0].transAxes, 
             size=fontsize)

# Second plot
axarr[0,1].plot(plottime2, random_walk2)
axarr[0,1].set_xlim([0, 2])
axarr[0,1].set_ylabel('Evidence', fontsize=fontsize, labelpad=-20)
axarr[0,1].set_yticks(np.array([0, 2.4]))
axarr[0,1].text(0.01, 0.8, '$\\delta$=3.0',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[0,1].transAxes, 
             size=fontsize)
axarr[0,1].text(0.01, 0.675, '$\\alpha$=2.4',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[0,1].transAxes, 
             size=fontsize)
axarr[0,1].text(0.01, 0.55, '$\\varsigma$=2.0',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[0,1].transAxes, 
             size=fontsize)
axarr[0,1].text(0.01, 0.325, '$\\delta / \\varsigma $=1.5',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[0,1].transAxes, 
             size=fontsize)
axarr[0,1].text(0.01, 0.2, '$\\alpha / \\varsigma $=1.2',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[0,1].transAxes, 
             size=fontsize)
axarr[0,1].text(-0.12, 0.97, 'b', weight='bold',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[0,1].transAxes, 
             size=fontsize)

if extra_row:
	# Third plot
	axarr[2,0].plot(plottime3, random_walk3)
	axarr[2,0].set_xlim([0, 2])
	axarr[2,0].set_ylabel('Evidence', fontsize=fontsize, labelpad=-20)
	axarr[2,0].set_yticks(np.array([0.0, 1.0]))
	axarr[2,0].set_yticklabels(np.array([0.0, 1.0]))
	axarr[2,0].text(0.01, 0.8, '$\\delta$=1.25',
	             horizontalalignment='left',
	             verticalalignment='center',
	             transform=axarr[2,0].transAxes, 
	             size=fontsize)
	axarr[2,0].text(0.01, 0.675, '$\\alpha$=1.0',
	             horizontalalignment='left',
	             verticalalignment='center',
	             transform=axarr[2,0].transAxes, 
	             size=fontsize)
	axarr[2,0].text(0.01, 0.55, '$\\varsigma=0.8\\bar{3}$',
	             horizontalalignment='left',
	             verticalalignment='center',
	             transform=axarr[2,0].transAxes, 
	             size=fontsize)
	axarr[2,0].text(0.01, 0.325, '$\\delta / \\varsigma $=1.5',
	             horizontalalignment='left',
	             verticalalignment='center',
	             transform=axarr[2,0].transAxes, 
	             size=fontsize)
	axarr[2,0].text(0.01, 0.2, '$\\alpha / \\varsigma $=1.2',
	             horizontalalignment='left',
	             verticalalignment='center',
	             transform=axarr[2,0].transAxes, 
	             size=fontsize)
	axarr[2,0].text(-0.12, 0.97, 'c', weight='bold',
	             horizontalalignment='left',
	             verticalalignment='center',
	             transform=axarr[2,0].transAxes, 
	             size=fontsize)

# First plot, second column
axarr[1,0].plot(plottime7, random_walk7)
axarr[1,0].set_xlim([0, 2])
#axarr[1,0].set_ylim([-0.2, 1.4])
axarr[1,0].set_ylabel('Evidence', fontsize=fontsize, labelpad=-20)
axarr[1,0].set_yticks(np.array([0, 1.2]))
axarr[1,0].text(0.01, 0.8, '$\\delta$=1.5',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[1,0].transAxes, 
             size=fontsize)
axarr[1,0].text(0.01, 0.675, '$\\alpha$=1.2',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[1,0].transAxes, 
             size=fontsize)
axarr[1,0].text(0.01, 0.55, '$\\varsigma$=0.5',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[1,0].transAxes, 
             size=fontsize)
axarr[1,0].text(0.01, 0.325, '$\\delta / \\varsigma $=3.0',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[1,0].transAxes, 
             size=fontsize)
axarr[1,0].text(0.01, 0.2, '$\\alpha / \\varsigma $=2.4',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[1,0].transAxes, 
             size=fontsize)
if extra_row:
	this_letter = 'd'
else:
	this_letter = 'c'
axarr[1,0].text(-0.12, 0.97, this_letter, weight='bold',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[1,0].transAxes, 
             size=fontsize)

# Second plot, second column
axarr[1,1].plot(plottime5, random_walk5)
axarr[1,1].set_xlim([0, 2])
axarr[1,1].set_ylabel('Evidence', fontsize=fontsize, labelpad=-20)
axarr[1,1].set_yticks(np.array([0, 1.2]))
axarr[1,1].text(0.01, 0.8, '$\\delta$=3.0',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[1,1].transAxes, 
             size=fontsize)
axarr[1,1].text(0.01, 0.675, '$\\alpha$=1.2',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[1,1].transAxes, 
             size=fontsize)
axarr[1,1].text(0.01, 0.55, '$\\varsigma$=2.0',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[1,1].transAxes, 
             size=fontsize)
axarr[1,1].text(0.01, 0.325, '$\\delta / \\varsigma $=1.5',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[1,1].transAxes, 
             size=fontsize)
axarr[1,1].text(0.01, 0.2, '$\\alpha / \\varsigma $=0.6',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[1,1].transAxes, 
             size=fontsize)
if extra_row:
	this_letter = 'e'
else:
	this_letter = 'd'
axarr[1,1].text(-0.12, 0.97, this_letter, weight='bold',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axarr[1,1].transAxes, 
             size=fontsize)

if extra_row:
	# Third plot, second column
	axarr[2,1].plot(plottime6, random_walk6)
	axarr[2,1].set_xlim([0, 2])
	axarr[2,1].set_ylabel('Evidence', fontsize=fontsize, labelpad=-20)
	axarr[2,1].set_yticks(np.array([0, 1.2]))
	axarr[2,1].text(0.01, 0.8, '$\\delta$=0.75',
	             horizontalalignment='left',
	             verticalalignment='center',
	             transform=axarr[2,1].transAxes, 
	             size=fontsize)
	axarr[2,1].text(0.01, 0.675, '$\\alpha$=1.2',
	             horizontalalignment='left',
	             verticalalignment='center',
	             transform=axarr[2,1].transAxes, 
	             size=fontsize)
	axarr[2,1].text(0.01, 0.55, '$\\varsigma$=0.5',
	             horizontalalignment='left',
	             verticalalignment='center',
	             transform=axarr[2,1].transAxes, 
	             size=fontsize)
	axarr[2,1].text(0.01, 0.325, '$\\delta / \\varsigma $=1.5',
	             horizontalalignment='left',
	             verticalalignment='center',
	             transform=axarr[2,1].transAxes, 
	             size=fontsize)
	axarr[2,1].text(0.01, 0.2, '$\\alpha / \\varsigma $=2.4',
	             horizontalalignment='left',
	             verticalalignment='center',
	             transform=axarr[2,1].transAxes, 
	             size=fontsize)
	axarr[2,1].text(-0.12, 0.97, 'f', weight='bold',
	             horizontalalignment='left',
	             verticalalignment='center',
	             transform=axarr[2,1].transAxes, 
	             size=fontsize)


# Make the x-axis labels for only the last x-axis in a column
axarr[rows-1, 1].set_xlabel('Time (secs)', fontsize=fontsize)
axarr[rows-1, 0].set_xlabel('Time (secs)', fontsize=fontsize)

# Set up the sharing of x-axes only within the same column
for i in range(rows):
    for j in range(columns):
        axarr[i, j].tick_params(axis='both', labelsize=fontsize)
        if i != rows - 1:  # Not the bottom row
            axarr[i, j].xaxis.set_tick_params(which='both', labelbottom=False, bottom=False)  # Remove ticks and labels from x-axis

# Tight layout with no space between plots
f.tight_layout(h_pad=0, w_pad=0)

plt.savefig("basic_DDMdc_simulations.png", dpi=300)