import math
import os
import random
import sys
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nengo
from nengo.processes import WhiteSignal
from nengo.processes import PresentInput
import pandas as pd
import scipy.signal as sig
import scipy.stats as stats
import yaml
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from nengo.processes import Piecewise, WhiteSignal
from nengo.utils.matplotlib import rasterplot
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.linear_model import LinearRegression
from matplotlib.gridspec import GridSpec
import networkx as nx
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.linalg import lstsq
from scipy.integrate import quad
import numpy as np
from nengo.solvers import LstsqL2
import nengo_bio as bio
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

# generate the mixture input
trial_num = 30 # number of trials
trial_interval = 0 # the time between each trial (seconds)
trial_duration = 5 # the time that each trial lasts (seconds)
electrode_number = 64 # total number of stimulation sites
source_number = 4 # number of inputs we're trying to seperate from
electrode_sample = np.floor(electrode_number / source_number) # how many stimulation sites are associated with each source (can overlap)

# model parameters
n_neurons = 100 # number of neurons in the ensemble
t_bin = 100 # size of spike count bins (ms)
dt=0.001 # time step (ms), so each point is a ms

type = "Balanced" # "Unbalanced" or "Balanced"
lesion_bars = False # True or False
e_to_i = 0.7
num_exc = int(n_neurons * e_to_i)
num_inh = n_neurons - num_exc

ablation_frac = 0.9 # fraction of neurons to ablate
dimensions = electrode_number
learning_rate = 1e-4
presentation_time = 1 # how long each stimulus is presented (seconds)
sim_time = trial_num * (trial_duration + trial_interval)

time_cut_off = (sim_time * 1000) # how many time steps to plot for the spike raster
testing_size = 0.3 # size of the testing set (fraction)

### Stochastic mixture stimulus over electrode subset
# Described in: Isomura and Friston, 2018, Scientific Reports
# 2D grid, treated as flattened array here
# trial duration and trial_is in units of timestamp (200 * dt = 200ms)
# source_number is the number of perceptual inputs we're trying to seperate from
# the generator_probability is the probability that an electrode receives input from a source (conditioned on at least one of the sources corresponding to it is on)
def stochastic_mixture(trial_num, trial_duration, trial_interval, 
    electrode_number, electrode_sample, num_sources, source_probability = 0.50, generator_probability = 0.75):    
    
    rng = np.random.default_rng()
    tsteps = int(trial_num*(trial_duration+trial_interval)) # ✅ total number of time steps

    input_stim_sites = np.zeros((num_sources, electrode_sample)) # ✅ each row is associated with one of the inputs (similar to how the halves were before)
    source_activity = np.zeros((num_sources, tsteps)) # ✅ 2d array where each row represents if the corresponding source is stimulating the group of electrodes at a given time step

    generator_data = np.random.binomial(1, generator_probability, size=(trial_num, electrode_number)) # ✅ whether or not the electrode can accept stimulus from the source at a given time interval
    zero_data = np.zeros((trial_num, electrode_number))

    for i in range(num_sources):
        input_stim_sites[i,:] = rng.choice(electrode_number, electrode_sample, replace=False) # ✅ associates electrodes to sources by picking electrode_sample number of electrodes from the electrode_number for each source
    
    input_stim_sites = input_stim_sites.astype(int) # convert to int for indexing
    source_activity = source_activity.astype(int) # convert to int for indexing

    mixture = np.zeros((trial_num*(trial_duration+trial_interval),electrode_number)) # this is the output of the entire electrode array

    counter_trial = 0
    for t in range(tsteps):
        if t % int(trial_duration+trial_interval) == 0: # wait until the start of a new trial
            for i in range(num_sources):
                if np.random.binomial(1,source_probability) == 1: 
                    mixture[t:t+trial_duration,input_stim_sites[i,:]] = generator_data[counter_trial,input_stim_sites[i,:]]
                    source_activity[i,t:t+trial_duration] = 1 # this source was on for this input
                else:
                    mixture[t:t+trial_duration,input_stim_sites[i,:]] = zero_data[counter_trial,input_stim_sites[i,:]]
                    source_activity[i,t:t+trial_duration] = 0 # this source was not on for this input
            mixture[t+trial_duration:t+trial_duration+trial_interval,:] = np.zeros((electrode_number,))
            counter_trial += 1

    return mixture, source_activity, input_stim_sites

# NOTE: this function generates a heatmap of the input to the mixture model, 
# while labeling which input dimensions correspond to which halves of the mixture model
def plot_mixture_input_heatmap(mixture_input, source_activity):
        plt.figure(figsize=(12, 8))

        group_masks = np.zeros((source_number, sim_time, electrode_number + source_number), dtype=bool) # equivalent to half1 and half2 masks from before
        source_masks = np.zeros((source_number, sim_time, electrode_number + source_number), dtype=bool) # equivalent to half1_source and half2_source masks from before
        for i in range(source_number):
            group_masks[i, :, input_stim_sites[i]] = True
            source_masks[i, :, electrode_number + i] = True
            mixture_input = np.concatenate((mixture_input, source_activity[i].reshape(-1, 1)), axis=1) # add stimulation site data to the the mixture inputs

        cmap = mcolors.LinearSegmentedColormap.from_list("", ["yellow", "green"]) # yellow represents off, green represents on
        
        # overlay the masks onto the data output
        input_source_colors = [
            "#000000",  # Black
            "#FF0000",  # Red
            "#0000FF",  # Blue
            "#800000",  # Maroon
            "#800080",  # Purple
            "#008080",  # Teal
            "#FFA500",  # Orange
            "#A52A2A",  # Brown
            "#FF00FF",  # Magenta
            "#FFC0CB"   # Pink
        ]

        for i in range(source_number):
            sns.heatmap(mixture_input.T, mask=~(group_masks[i]).T, cmap=sns.light_palette(input_source_colors[i], as_cmap=True), cbar=False, alpha=0.3)
            sns.heatmap(mixture_input.T, mask=~(source_masks[i]).T, cmap=cmap, cbar=False, alpha=0.3)

        # create custom patches for the inputs
        patches = []
        for i in range(source_number):
            patches.append(mpatches.Patch(color=input_source_colors[i], label=f'Group {i+1}'))
        
        patch_source_on = mpatches.Patch(color="green", label='Source On')
        patches.append(patch_source_on)
        patch_source_off = mpatches.Patch(color="yellow", label='Source Off')
        patches.append(patch_source_off)
        
        # Add the legend to the plot
        plt.legend(handles=patches, loc='upper right')
        plt.xlabel('Time Steps (Seconds)')
        plt.ylabel('Electrode')
        plt.title('Mixture Input Over Time')
        plt.xticks(np.arange(0, len(mixture_input)+1, 5), labels=np.arange(0, len(mixture_input)+1, 5))
        plt.show()
        # plt.savefig(path + "/mixture_input.png", dpi=300)

mixture, source_activity, input_stim_sites = stochastic_mixture(trial_num=trial_num, trial_duration=trial_duration, 
                             trial_interval=trial_interval, electrode_number=electrode_number, num_sources=source_number,
                             electrode_sample=electrode_sample)

# plot the mixture input
plot_mixture_input_heatmap(mixture, source_activity)