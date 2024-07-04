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
import numpy as np
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
from sklearn.preprocessing import StandardScaler
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
import nengo
from nengo.processes import WhiteSignal
from nengo.processes import PresentInput
from nengo.solvers import LstsqL2
from sklearn.preprocessing import StandardScaler
from nengo.utils.matplotlib import rasterplot
from sklearn.decomposition import PCA
import nengo_bio as bio

os.environ
OUT_DIR = "/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-01-12 Classification" #NOTE: change this for output folder
os.system('mkdir {}'.format(os.path.join(OUT_DIR,'plots')))

name = "Attempt 1"

path = os.path.join(OUT_DIR,f'{name} plots')
if not os.path.exists(path):
        os.makedirs(path)

def stochastic_mixture(trial_num = 256, trial_duration=200, trial_interval=1000, 
    source_number = 64, source_sample = 32, source_probability = 0.75, generator_probability = 0.5):    
    """
    Stochastic mixture stimulus over electrode subset 
    # Described in: Isomura and Friston, 2018, Scientific Reports
    # 2D grid, treated as flattened array here 
    # trial duration and trial_is in units of timestamp (200 * dt = 200ms)
    # 16 of 32 electrodes were stim'd under source 1, with a prob 3/4, or source 2, with a prob 1/4. 
    # Conversely, the other 16  were stim'd under source 1, with a prob 1/4, or source 2, with a prob of 3/4 
    # A session of training comprising 256 trials with the 1 s intervals, followed by 244 second rest periods. We repeated this training 500 second cycle for 100 sessions.
    # The 32 stimulated sites are randomly selected in advance from an 8×8 grid.
    """
    rng = np.random.default_rng()
    source_subsample = int(source_sample/2) # 16 channels split between 32 sampled for the two source
    half1 = rng.choice(source_number, source_subsample, replace=False)
    half2 = rng.choice(source_number, source_subsample, replace=False)
    
    #source1 = np.random.binomial(1,0.5, size=(trial_num*(trial_duration+trial_interval),source_subsample)) 
    #source2 = np.random.binomial(1,0.5, size=(trial_num*(trial_duration+trial_interval),source_subsample))
    source1 = np.random.binomial(1,generator_probability, size=(trial_num,source_subsample)) # for the first trial
    source2 = np.random.binomial(1,generator_probability, size=(trial_num,source_subsample))
    
    tsteps = int(trial_num*(trial_duration+trial_interval))
    mixture = np.zeros((trial_num*(trial_duration+trial_interval),source_number))
    counter_trial = 0

    half_1_source = np.zeros(trial_num*(trial_duration+trial_interval))
    half_2_source = np.zeros(trial_num*(trial_duration+trial_interval))

    for t in range(tsteps):
        if t % int(trial_duration+trial_interval) == 0:
            
            if np.random.binomial(1,source_probability) == 1: 
                # s1 to h1
                mixture[t:t+trial_duration,half1] = source1[counter_trial]
                half_1_source[t:t+trial_duration] = 0
            else:
                # s2 to h1
                mixture[t:t+trial_duration,half1] = source2[counter_trial]
                half_1_source[t:t+trial_duration] = 1
            if np.random.binomial(1,source_probability) == 1:
                # s2 to h2
                mixture[t:t+trial_duration,half2] = source2[counter_trial]
                half_2_source[t:t+trial_duration] = 1
            else:
                # s1 to h2
                mixture[t:t+trial_duration,half2] = source1[counter_trial]
                half_2_source[t:t+trial_duration] = 0
            mixture[t+trial_duration:t+trial_duration+trial_interval,:] = np.zeros((source_number,))
            counter_trial += 1

    return mixture, half_1_source, half_2_source, half1, half2

# NOTE: population-level ablation (has the capability to lesion across neuronal ensembles)
# inputs: vector of ensembles we want to lesion, proportion of overall neurons that should be lesioned
# outputs: none, but affects n neurons in each ensemble such that their encoder values are 0 and bias terms are -1000
# -- ensembles of form [exc_pop, inh_pop, .........]
# -- proportion: fraction of neurons that we want to ablate
def ablate_population(ensembles, proportion, sim, bias=True):
    pop_neurons = 0 # compute the total # of neurons in the population
    for ens in ensembles: pop_neurons = pop_neurons + ens.n_neurons

    n_lesioned =  min(int(pop_neurons * proportion), pop_neurons) # ensure that prop ablated doesn't exceed total
    #print(f"n_lesioned = {n_lesioned}, proportion = {proportion}")

    # select the neurons we're going to ablate (simple random sampling via a lottery)
    losers = np.sort(np.random.choice(np.arange(pop_neurons), replace=False, size=n_lesioned)) # these are the neurons we want to ablate
    #print(f"losers = {losers}")

    group_idx = 0 # keep track of where we are relative to ensembles
    for ens in ensembles: # loop over the ensembles
        #print(f"group_idx = {group_idx}")
        for loser in losers:
            if group_idx <= loser < (group_idx + ens.n_neurons): # avoid double_lesioning, lower inclusive to capute start of populations
                rel_neur_idx = loser - group_idx
                #print(f"rel_neur_idx = {rel_neur_idx}")
                #print(f"ensemble neurons = {ens.n_neurons}")

                encoder_sig = sim.signals[sim.model.sig[ens]["encoders"]]
                encoder_sig.setflags(write=True)
                encoder_sig[rel_neur_idx] = 0.0 # the encoders (linear weighting of output populations) are set to 0
                encoder_sig.setflags(write=False)

                if bias:
                    bias_sig = sim.signals[sim.model.sig[ens.neurons]["bias"]]
                    bias_sig.setflags(write=True)
                    bias_sig[rel_neur_idx] = -1000000000000 # the bias term is set to 0, making the populations un-excitable
            
        group_idx = group_idx + ens.n_neurons
    return losers

# is designed to cycle through the elements of an array x based on a given time t
def cycle_array(x, period, dt=0.001):
    """Cycles through the elements"""
    i_every = int(round(period / dt))
    if i_every != period / dt:
        raise ValueError(f"dt ({dt}) does not divide period ({period})")

    def f(t):
        i = int(round((t - dt) / dt))  # t starts at dt
        return x[int(i / i_every) % len(x)]

    return f

# NOTE: this function generates a heatmap of the input to the mixture model, while labeling which neurons correspond to
# which halves of the mixture model
def plot_mixture_input_heatmap(mixture_input, half1, half2, half1_source, half2_source):
        plt.figure(figsize=(12, 8))

        # Create masks for half1 and half2 rows
        mask_half1 = np.zeros((num_items,source_number + 2), dtype=bool)
        mask_half1[:, half1] = True

        mask_half2 = np.zeros((num_items,source_number + 2), dtype=bool)
        mask_half2[:, half2] = True

        mask_half1_source = np.zeros((num_items,source_number + 2), dtype=bool)
        mask_half1_source[:, source_number] = True

        mask_half2_source = np.zeros((num_items,source_number + 2), dtype=bool)
        mask_half2_source[:, source_number + 1] = True

        half1_source = half1_source.reshape(-1, 1)
        half2_source = half2_source.reshape(-1, 1)

        # print(f"half1source {half1_source.shape}")
        # print(f"half2source {half2_source.shape}")
        # print(f"mixture {mixture_input.shape}")
        # print(f"half1 {half1}")
        # print(f"half2 {half2}")
        # print(f"half1source {half1_source.T}")
        # print(f"half2source {half2_source.T}")

        # Concatenate half1_source and half2_source at the bottom
        mixture_input_extended = np.concatenate((mixture_input, half1_source, half2_source), axis=1)

        cmap = mcolors.LinearSegmentedColormap.from_list("", ["yellow", "red"]) # yellow represents source1, red represents source2

        # Overlay colored masks
        sns.heatmap(mixture_input_extended.T, mask=~mask_half2.T, cmap=sns.light_palette("green", as_cmap=True), cbar=False, alpha=0.3)
        sns.heatmap(mixture_input_extended.T, mask=~mask_half1.T, cmap=sns.light_palette("blue", as_cmap=True), cbar=False, alpha=0.3)
        sns.heatmap(mixture_input_extended.T, mask=~mask_half2_source.T, cmap=cmap, alpha=0.3, cbar=False)
        sns.heatmap(mixture_input_extended.T, mask=~mask_half1_source.T, cmap=cmap, alpha=0.3, cbar=False)

        # Create custom patches as handles for the legend
        patch_half1 = mpatches.Patch(color=sns.light_palette("blue", as_cmap=True)(0.5), label='Half 1')
        patch_half2 = mpatches.Patch(color=sns.light_palette("green", as_cmap=True)(0.5), label='Half 2')
        patch_half2_source = mpatches.Patch(color="yellow", label='Source 1 Used')
        patch_half1_source = mpatches.Patch(color="red", label='Source 2 Used')
        
        # Add the legend to the plot
        plt.legend(handles=[patch_half1, patch_half2, patch_half1_source, patch_half2_source], loc='upper right')

        plt.xlabel('Time Steps')
        plt.ylabel('Electrode')
        plt.title('Mixture Input Over Time')
        plt.savefig(path + "/mixture_input.png", dpi=300)

# generate the mixture input
trial_num = 10
trial_duration = 50 
trial_interval = 0 # the time between each trial
source_number = 30 # total number of sources
source_sample = 30 # how many sources we subsample from the total number of sources

mixture_input, half1_source, half2_source, half1, half2 = stochastic_mixture(trial_num=trial_num, trial_duration=trial_duration, 
                             trial_interval=trial_interval, source_number=source_number, 
                             source_sample=source_sample)

print(f"mixture_input: {mixture_input}, with shape: {np.shape(mixture_input)}")
print(f"half 1 mixtrue input: {mixture_input[:,half1]}, with shape: {np.shape(mixture_input[:,half1])}")
print(f"half1_source: {half1_source}, with shape: {np.shape(half1_source)}")
print(f"half2_source: {half2_source}, with shape: {np.shape(half2_source)}")
print(f"half1: {half1}, with shape: {np.shape(half1)}")
print(f"half2: {half2}, with shape: {np.shape(half2)}")

# setting up the voja learning rules
num_items = trial_num * (trial_duration + trial_interval)
d_key = len(half1)
d_value = 1 # the value is a scalar, since we're trying to differentiate between a zero (source 1) and one (source 2)

keys = (mixture_input[:,half1]) / 4 # grab only the inputs going to half 1, and normalize them so that the intercepts are less than 1
values = half1_source / 4 # grab only the inputs going to half 1, and normalize them so that the intercepts are less than 1

# An important quantity is the largest dot-product between all pairs of keys, since a neuron’s intercept should 
# not go below this value if it’s positioned between these two keys. Otherwise, the neuron will move back and 
# forth between encoding those two inputs.
# https://www.nengo.ai/nengo/examples/learning/learn-associations.html
intercept = (np.dot(keys, keys.T) - np.eye(num_items)).flatten().max() # find the largest intercept
print(f"intercept: {intercept}")

# Model constants
n_neurons = 100
dt = 0.001
period = 1 # this is the amount of time that each item is presented, clamp at 1
T = period * num_items * 2 # we multiply by 2 such that we can present the items twice (once for learning, once for recall)

plot_mixture_input_heatmap(mixture_input, half1, half2, half1_source, half2_source) # plot the mixture input

# # NOTE: population-level ablation (has the capability to lesion across neuronal ensembles)
# type = "Learning" # "Unbalanced" or "Balanced" or "Learning"
# e_to_i = 0.7
# num_exc = int(n_neurons * e_to_i)
# num_inh = n_neurons - num_exc
ablation_frac = 0.5 # the fraction of neurons to ablate

# Model network
model = nengo.Network()
with model:
    # Create the inputs/outputs
    stim_keys = nengo.Node(output=cycle_array(keys, period, dt))
    stim_values = nengo.Node(output=cycle_array(values, period, dt))
    learning = nengo.Node(output=lambda t: -int(t >= T / 2))
    recall = nengo.Node(size_in=d_value)

    # Create the memory -- this is the node of interest that we can modulate
    memory = nengo.Ensemble(n_neurons, d_key, intercepts=[intercept] * n_neurons)

    # Learn the encoders/keys
    voja = nengo.Voja(learning_rate=5e-3, post_synapse=None)
    conn_in = nengo.Connection(stim_keys, memory, synapse=None, learning_rule_type=voja)
    nengo.Connection(learning, conn_in.learning_rule, synapse=None)

    # Learn the decoders/values, initialized to a null function
    conn_out = nengo.Connection(
        memory,
        recall,
        learning_rule_type=nengo.PES(1e-3),
        function=lambda x: np.zeros(d_value),
    )

    # Create the error population
    error = nengo.Ensemble(n_neurons, d_value)
    nengo.Connection(
        learning, error.neurons, transform=[[10.0]] * n_neurons, synapse=None # weight it much more heavily than the initial population
    )

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(stim_values, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    # Setup probes
    p_keys = nengo.Probe(stim_keys, synapse=None)
    p_values = nengo.Probe(stim_values, synapse=None)
    p_learning = nengo.Probe(learning, synapse=None)
    p_error = nengo.Probe(error, synapse=0.005)
    p_recall = nengo.Probe(recall, synapse=None)
    p_encoders = nengo.Probe(conn_in.learning_rule, "scaled_encoders")

    mem_spikes = nengo.Probe(memory.neurons)

with nengo.Simulator(model, dt=dt) as sim:
    sim.run(T)
    t = sim.trange()

    plt.figure()
    plt.title("Keys")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.plot(t, sim.data[p_keys])
    plt.ylim(-0.1, 0.6)
    plt.savefig(path + "/pre_keys.png", dpi= 300)

    plt.figure()
    plt.title("Values")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.plot(t, sim.data[p_values])
    plt.ylim(-0.1, 0.6)
    plt.savefig(path + "/pre_values.png", dpi= 300)

    plt.figure()
    plt.title("Learning")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.plot(t, sim.data[p_learning])
    plt.ylim(-1.1, 0.1)
    plt.savefig(path + "/pre_learning.png", dpi= 300)

    plt.figure()
    plt.title("Recall")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.plot(t, sim.data[p_recall])
    plt.savefig(path + "/pre_recall.png", dpi= 300)

    train = t <= T / 2
    test = ~train

    plt.figure()
    plt.title("Value Error During Training")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.plot(t[train], sim.data[p_error][train])
    plt.savefig(path + "/pre_error_train.png", dpi= 300)

    plt.figure()
    plt.title("Value Error During Recall")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.plot(t[test], sim.data[p_recall][test] - sim.data[p_values][test])
    plt.savefig(path + "/pre_error_recall.png", dpi= 300)

    recall_error = np.abs(sim.data[p_recall][test] - sim.data[p_values][test]) # compute the overall recall error during the test phase
    average_recall_error = np.mean(recall_error) # use this as a performance metric of interest

    # ----------------------------------------------------------------------------------------------#
    # spike raster visualization
    fig, ax = plt.subplots()
    mem_spikes = sim.data[mem_spikes][10:, :] # spiking at each milisecond -- (length of sim (ms) x num_neurons)
    binary = mem_spikes.T # (num_neurons x length of sim (ms))
    spike_rates = np.sum(binary, axis=1) / (T / n_neurons) # (num_neurons x 1) -- average firing rate of each neuron
    for neuron_idx in range(n_neurons):
        spike_times = np.where(binary[neuron_idx])[0] # get the spike times for each neuron
        ax.scatter(spike_times, np.full_like(spike_times, neuron_idx), marker='|', color = 'green', s=0.5) # plot the spike times for each neuron
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron')
    ax.set_title('Spiking Activity', fontsize=14, fontweight='bold')
    plt.savefig(path + "/spike_raster.png", dpi = 300)

    # ----------------------------------------------------------------------------------------------#

    # NOTE: population-level ablation (has the capability to lesion across neuronal ensembles)
    ablate_population([memory], ablation_frac, sim)

    # ----------------------------------------------------------------------------------------------#

    # rerun the simulation with the ablated population
    # sim.run(T)


fig, axs = plt.subplots(2, 3, figsize=(18,10))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure
axs[0].imshow(plt.imread(path + "/pre_keys.png"), aspect='auto')
axs[1].imshow(plt.imread(path + "/pre_values.png"), aspect='auto')
axs[2].imshow(plt.imread(path + "/pre_recall.png"), aspect='auto')
axs[3].imshow(plt.imread(path + "/pre_error_train.png"), aspect='auto')
axs[4].imshow(plt.imread(path + "/pre_error_recall.png"), aspect='auto')
axs[5].imshow(plt.imread(path + "/mixture_input.png"), aspect='auto')
fig.suptitle(f'{name}, encoder/decoder mapping, average recall error {average_recall_error:.3f}', fontsize = 10)
fig.savefig(path + "/pre_mapping_fig", dpi = 300)
plt.tight_layout()

# ----------------------------------------------------------------------------------------------#

# # examining encoder changes
# scale = (sim.data[memory].gain / memory.radius)[:, np.newaxis]

# def plot_2d(text, xy):
#     plt.figure()
#     plt.title(text)
#     plt.scatter(xy[:, 0], xy[:, 1], label="Encoders")
#     plt.scatter(keys[:, 0], keys[:, 1], c="red", s=150, alpha=0.6, label="Keys")
#     plt.xlim(-1.5, 1.5)
#     plt.ylim(-1.5, 2)
#     plt.legend()
#     plt.gca().set_aspect("equal")

# plot_2d("Before", sim.data[p_encoders][0].copy() / scale)
# plot_2d("After", sim.data[p_encoders][-1].copy() / scale)

# ----------------------------------------------------------------------------------------------- #

##### NOTE: display/close functionality for plotting ease #####
plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run

# ----------------------------------------------------------------------------------------------- #