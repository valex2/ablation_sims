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
# from bilm import bilm
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
OUT_DIR = "/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/12:1:23 Input Selection" #NOTE: change this for output folder
os.system('mkdir {}'.format(os.path.join(OUT_DIR,'plots')))

name = "Testing"

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
    print(np.shape(source1))
    print(np.shape(source2))
    print(f"sourc1: {source1}, source2: {source2}")
    
    tsteps = int(trial_num*(trial_duration+trial_interval))
    mixture = np.zeros((trial_num*(trial_duration+trial_interval),source_number))
    counter_trial = 0

    print(f"half1: {half1}, half2: {half2}")
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

    print(f"half_1_source: {half_1_source}, len: {len(half_1_source)}")
    print(f"half_2_source: {half_2_source}, len: {len(half_2_source)}")

    print(mixture)
    print(f"mixture shape: {np.shape(mixture)}")
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

### generate stimulus sequence
n_neurons = 100

balance = True
e_to_i = 0.7
num_exc = int(n_neurons * e_to_i)
num_inh = n_neurons - num_exc

ablation_frac = 0.9
source_number = 10
source_sample = 10
dimensions = source_number
learning_rate = 1e-4
trial_num = 10
trial_interval = 3
trial_duration = 5
presentation_time = (2*(trial_duration))/10
sim_time = trial_num * (trial_duration + trial_interval)
mixture_input, half1_source, half2_source, half1, half2 = stochastic_mixture(trial_num=trial_num, trial_duration=trial_duration, 
                             trial_interval=trial_interval, source_number=source_number, 
                             source_sample=source_sample)


model = nengo.Network(label="Predicting Mixture Input")
with model:
    if (balance == False):
        # Input Node - PresentInput can be used to present stimulus in time sequence
        stim = nengo.Node(PresentInput(mixture_input, presentation_time= presentation_time))

        # Ensemble
        ensemble = nengo.Ensemble(n_neurons=n_neurons, dimensions=dimensions, neuron_type=nengo.Izhikevich(tau_recovery=0.15))

        # Output Node
        output = nengo.Node(size_in=dimensions)

        # Error Node
        error = nengo.Node(size_in=dimensions)

        # Learning connections
        conn = nengo.Connection(ensemble, output, function=lambda x: [0]*dimensions, 
                                learning_rule_type=nengo.PES(learning_rate))

        # Feedback for learning
        nengo.Connection(output, error)
        nengo.Connection(stim, error, transform=-1)
        nengo.Connection(error, conn.learning_rule)

        # Input and Ensemble connections
        nengo.Connection(stim, ensemble)

        # Probe to monitor output
        output_probe = nengo.Probe(output, synapse=0.01)
        input_probe = nengo.Probe(stim, synapse=0.01)
        error_probe = nengo.Probe(error, synapse=0.01)
    else:
        stim = nengo.Node(PresentInput(mixture_input, presentation_time= presentation_time))

        # Ensembles
        exc_pop = nengo.Ensemble(n_neurons=num_exc, dimensions=dimensions, neuron_type=nengo.Izhikevich(tau_recovery=0.15))
        inh_pop = nengo.Ensemble(n_neurons=num_inh, dimensions=dimensions)

        output = nengo.Node(size_in=dimensions)
        error = nengo.Node(size_in=dimensions)
        
        nengo.Connection(stim, exc_pop) #input to excitatory
        nengo.Connection(exc_pop, inh_pop, transform=1) #excitatory to inhibitory
        nengo.Connection(inh_pop, exc_pop, transform=-1) #inhibitory to excitatory

        conn = nengo.Connection(exc_pop, output, function=lambda x: [0]*dimensions,
                                learning_rule_type=nengo.PES(learning_rate))
        nengo.Connection(output, error)
        nengo.Connection(stim, error, transform=-1)
        nengo.Connection(error, conn.learning_rule)

        output_probe = nengo.Probe(output, synapse=0.01)
        input_probe = nengo.Probe(stim, synapse=0.01)
        error_probe = nengo.Probe(error, synapse=0.01)

# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(sim_time)

    if (balance == False):
        ablate_population([ensemble], ablation_frac, sim)
    else:
        ablate_population([exc_pop, inh_pop], ablation_frac, sim)

    # Plot output
    plt.figure(figsize=(12, 6))
    plt.plot(sim.trange(), sim.data[output_probe], label='Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Output Over Time')
    plt.legend()
    plt.savefig(path + "/pre_output.png", dpi= 300)

    # plot input 
    plt.figure(figsize=(12, 6))
    plt.plot(sim.trange(), sim.data[input_probe], label='Input', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Input Over Time')
    plt.legend()
    plt.savefig(path + "/pre_input.png", dpi= 300)

    # plot error
    plt.figure(figsize=(12, 6))
    plt.plot(sim.trange(), sim.data[error_probe], label='Error', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Error Over Time')
    plt.legend()
    plt.savefig(path + "/pre_error.png", dpi= 300)

    # def plot_mixture_input_heatmap(mixture_input):
    #     plt.figure(figsize=(12, 8))
    #     sns.heatmap(mixture_input.T, cmap="viridis", cbar_kws={'label': 'Stimulation Intensity'})
    #     plt.xlabel('Time Steps')
    #     plt.ylabel('Electrode')
    #     plt.title('Mixture Input Over Time')
    #     plt.savefig(path + "/mixture_input.png", dpi= 300)

    def plot_mixture_input_heatmap(mixture_input, half1, half2, half1_source, half2_source):
        plt.figure(figsize=(12, 8))

        # Create masks for half1 and half2 rows
        mask_half1 = np.zeros((80,source_number + 2), dtype=bool)
        mask_half1[:, half1] = True

        mask_half2 = np.zeros((80,source_number + 2), dtype=bool)
        mask_half2[:, half2] = True

        mask_half1_source = np.zeros((80,source_number + 2), dtype=bool)
        mask_half1_source[:, source_number] = True

        mask_half2_source = np.zeros((80,source_number + 2), dtype=bool)
        mask_half2_source[:, source_number + 1] = True

        half1_source = half1_source.reshape(-1, 1)
        half2_source = half2_source.reshape(-1, 1)

        print(f"half1source {half1_source.shape}")
        print(f"half2source {half2_source.shape}")
        print(f"mixture {mixture_input.shape}")
        print(f"half1 {half1}")
        print(f"half2 {half2}")
        print(f"half1source {half1_source.T}")
        print(f"half2source {half2_source.T}")

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

    plot_mixture_input_heatmap(mixture_input, half1, half2, half1_source, half2_source)

    # Extract actual inputs and model outputs
    actual_inputs = sim.data[input_probe]
    model_outputs = sim.data[output_probe]

    # Compute MSE
    pre_mse = ((actual_inputs - model_outputs) ** 2).mean()

    # Compute Correlation Coefficient
    pre_correlation_matrix = np.corrcoef(actual_inputs.ravel(), model_outputs.ravel())
    pre_correlation_coefficient = pre_correlation_matrix[0, 1]

    #--------------------------------------------------------------------------------------------------#

    sim.run(sim_time)
    plt.figure(figsize=(12, 6))
    plt.plot(sim.trange()[sim_time*1000 + 10:], sim.data[output_probe][sim_time*1000 + 10:, :], label='Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Output Over Time')
    plt.legend()
    plt.savefig(path + "/post_output.png", dpi= 300)    

    # plot input 
    plt.figure(figsize=(12, 6))
    plt.plot(sim.trange()[sim_time*1000 + 10:], sim.data[input_probe][sim_time*1000 + 10:, :], label='Input', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Input Over Time')
    plt.legend()
    plt.savefig(path + "/post_input.png", dpi= 300)

    # plot error
    plt.figure(figsize=(12, 6))
    plt.plot(sim.trange()[sim_time*1000 + 10:], sim.data[error_probe][sim_time*1000 + 10:, :], label='Error', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Error Over Time')
    plt.legend()
    plt.savefig(path + "/post_error.png", dpi= 300)

    # Extract actual inputs and model outputs
    actual_inputs = sim.data[input_probe][sim_time*1000 + 10:, :]
    model_outputs = sim.data[output_probe][sim_time*1000 + 10:, :]

    # Compute MSE
    post_mse = ((actual_inputs - model_outputs) ** 2).mean()

    # Compute Correlation Coefficient
    post_correlation_matrix = np.corrcoef(actual_inputs.ravel(), model_outputs.ravel())
    post_correlation_coefficient = post_correlation_matrix[0, 1]    

    #--------------------------------------------------------------------------------------------------#

fig, axs = plt.subplots(2, 2, figsize=(18,10))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure
axs[0].imshow(plt.imread(path + "/pre_input.png"), aspect='auto')
axs[1].imshow(plt.imread(path + "/pre_output.png"), aspect='auto')
axs[2].imshow(plt.imread(path + "/pre_error.png"), aspect='auto')
axs[3].imshow(plt.imread(path + "/mixture_input.png"), aspect='auto')

fig.suptitle(f'{name}, pre-ablation -- MSE = {pre_mse:.3f}, CorrelationCoeff = {pre_correlation_coefficient:.3f}', fontsize = 10)
fig.savefig(path + "/pre_overview_fig", dpi = 300)
plt.tight_layout()

fig, axs = plt.subplots(2, 2, figsize=(18,10))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure
axs[0].imshow(plt.imread(path + "/post_input.png"), aspect='auto')
axs[1].imshow(plt.imread(path + "/post_output.png"), aspect='auto')
axs[2].imshow(plt.imread(path + "/post_error.png"), aspect='auto')
axs[3].imshow(plt.imread(path + "/mixture_input.png"), aspect='auto')

fig.suptitle(f'{name}, post-ablation -- MSE = {post_mse:.3f}, CorrelationCoeff = {post_correlation_coefficient:.3f}', fontsize = 10)
fig.savefig(path + "/post_overview_fig", dpi = 300)
plt.tight_layout()

# ----------------------------------------------------------------------------------------------- #

##### NOTE: display/close functionality for plotting ease #####
plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run

