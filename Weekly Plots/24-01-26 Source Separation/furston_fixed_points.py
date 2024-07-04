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

os.environ
OUT_DIR = "/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-02-09 Source Attractors" #NOTE: change this for output folder
os.system('mkdir {}'.format(os.path.join(OUT_DIR,'plots')))

name = "Fixed Points (4 Sources) (90%)"

path = os.path.join(OUT_DIR,f'{name} plots')
if not os.path.exists(path):
        os.makedirs(path)

# generate the mixture input
trial_num = 50 # number of trials
trial_interval = 0 # the time between each trial (seconds)
trial_duration = 5 # the time that each trial lasts (seconds)
electrode_number = 64 # total number of stimulation sites
source_number = 4 # number of inputs we're trying to seperate from
electrode_sample = int(np.floor(electrode_number / source_number)) # how many stimulation sites are associated with each source (can overlap)

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
num_eigsum = 15 # number of eigenvalues to sum for the POV

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
        # plt.show()
        plt.savefig(path + "/mixture_input.png", dpi=300)

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

##### NOTE: given an array of data samples x features computes the plane of best fit in 3-space and returns the fit (3x1) and residual (1,)
def plane_of_best_fit(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    tmp_A = [] # create an overconstrained system of equations 
    tmp_b = []
    
    for i in range(len(x)):
        tmp_A.append([x[i], y[i], 1])
        tmp_b.append([z[i]])
    
    b = np.matrix(tmp_b) # 99x1
    A = np.matrix(tmp_A) # 99x3

    fit, residual, rank, s = lstsq(A, b) # solve the system -- solution is z = fit[0] * x + fit[1] * y + fit[2] 
    return fit, residual

##### NOTE: this function computes a measure of the spreadiness of different trajectories #####
def line_integral(data):
    squared_data = np.real(data) ** 2
    sum_across_features = np.sum(squared_data, axis=1)
    two_way_sum = np.log(np.sum(sum_across_features) / (len(data[0, :]) * len(data[:, 0])))
    return two_way_sum

def average_classification_performance(X, sources):
    """
    Computes the average classification performance across multiple sources.

    Parameters:
    X (array-like): The feature dataset.
    sources (list of array-like): List of source arrays.
    t_bin (int): Time bin size for repeating the sources.
    testing_size (float): The proportion of the dataset to include in the test split.
    """
    num_sources = len(sources)

    # Initializing the performance metrics
    total_performance = {
        "off": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "on": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "accuracy": 0
    }

    source_accuracy = []

    for source in sources:
        # Splitting the dataset into training and testing sets for each source
        y = np.repeat(source, (1000 / t_bin))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_size, random_state=42)

        # Initialize and train the Logistic Regression model
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = log_reg.predict(X_test)

        # Creating a classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        source_off = df_report.loc["0"]
        source_on = df_report.loc["1"]

        # Computing the performance metrics
        total_performance["off"]["precision"] += source_off["precision"]
        total_performance["off"]["recall"] += source_off["recall"]
        total_performance["off"]["f1-score"] += source_off["f1-score"]
        total_performance["off"]["support"] += source_off["support"]

        total_performance["on"]["precision"] += source_on["precision"]
        total_performance["on"]["recall"] += source_on["recall"]
        total_performance["on"]["f1-score"] += source_on["f1-score"]
        total_performance["on"]["support"] += source_on["support"]

        total_performance["accuracy"] += report["accuracy"]

        source_accuracy.append(report["accuracy"])


    # averaging the performance metrics
    total_performance["off"]["precision"] /= num_sources
    total_performance["off"]["recall"] /= num_sources
    total_performance["off"]["f1-score"] /= num_sources
    total_performance["off"]["support"] /= num_sources
    
    total_performance["on"]["precision"] /= num_sources
    total_performance["on"]["recall"] /= num_sources
    total_performance["on"]["f1-score"] /= num_sources
    total_performance["on"]["support"] /= num_sources

    total_performance["accuracy"] /= num_sources

    return total_performance, source_accuracy

# NOTE: given a vector of sorted eigenvalues, plots the POV by the top num_eigsum eigenvalues in histogram form
# also computes a spread metric for the eigenvalues, which is assentially a standardized measure of the variance of the population eigenvalues
# computed via dot-producting an increasing array of len num_eigsum by the data itself and then standardizing based on array length
# RETURNS: plot, metric
def plot_POV(data, modifier, num_eigs = num_eigsum):
    n = np.arange(len(data)) # (1 x num_eigsum)
    n = n.T # (num_eigsum x 1)
    spread_metric = (data @ n) / 100 # compute a standardized metric of the "spread" of the eigenvalues

    fig, ax = plt.subplots()
    x_pos = np.arange(num_eigs)
    bar_width = 0.8
    bars = ax.bar(x_pos, data[:num_eigs], color='lightgray', edgecolor='black', linewidth=1, width=bar_width, zorder=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'PC{i}' for i in range(num_eigs)], fontsize=10, rotation=90, ha='center')
    ax.set_ylabel('percentage of variance explained', fontsize=12)
    ax.set_title(f'Variance Explained by First {num_eigsum} PCs, {modifier}', fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    for bar in bars: # label the height of each bar with smaller font size
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom', fontsize=8)

    ax.text(0.5, 0.95, f'Standardized Spread: {spread_metric:.2f}', transform=ax.transAxes, ha='center', fontsize=12)  # Add the spread_metric value to the plot
    fig.tight_layout()
    return(fig, spread_metric)

# NOTE: given n vectors of sorted eigenvalues, plots the POVs by the top num_eigsum eigenvalues in histogram form (overlayed)
# also computes a spread metric for the eigenvalues, which is assentially a standardized measure of the variance of the population eigenvalues
# computed via dot-producting an increasing array of len num_eigsum by the data itself and then standardizing based on array length
# RETURNS: plot, (spread metric 1, spread metric 2)
# colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightpink', 'lightyellow']
def plot_POVs(data_list, modifier_list, colors, title, num_eigs=num_eigsum):
    bars_list = []
    spread_metrics = []

    n = np.arange(num_eigs)
    n = n.T
    fig, ax = plt.subplots()
    total_bars = len(data_list)
    
    max_bar_width = 0.4 
    available_space = max_bar_width * total_bars
    bar_width = available_space / (total_bars + 1)
    
    for i, (data, modifier, color) in enumerate(zip(data_list, modifier_list, colors)):
        spread_metric = (data[:num_eigsum] @ n) / 100
        spread_metrics.append(spread_metric)
        
        x_pos = np.arange(num_eigs) + i * bar_width  # Adjust x_pos for each group of bars
        bars = ax.bar(x_pos, data[:num_eigs], color=color, edgecolor='black', linewidth=1, width=bar_width, zorder=2, label=f"{modifier}, σ={spread_metric:.2f}")
        bars_list.append(bars)
    
    total_bars = len(data_list)
    x_pos_middle = np.arange(num_eigs) + (total_bars - 1) * bar_width / 2
    ax.set_xticks(x_pos_middle)
    ax.set_xticklabels([f'PC{i}' for i in range(num_eigs)], fontsize=10, rotation=90, ha='center')
    ax.set_ylabel('percentage of variance explained', fontsize=12)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    for bars, modifier, color in zip(bars_list, modifier_list, colors):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 4),
                        textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold', color=color, rotation = "vertical")
    
    fig.tight_layout()
    return fig, spread_metrics

###### -------------------------------------------------------------------------------------------------- ######

mixture, source_activity, input_stim_sites = stochastic_mixture(trial_num=trial_num, trial_duration=trial_duration, 
                             trial_interval=trial_interval, electrode_number=electrode_number, num_sources=source_number,
                             electrode_sample=electrode_sample)

# plot the mixture input
plot_mixture_input_heatmap(mixture, source_activity)

###### -------------------------------------------------------------------------------------------------- ######

# Model network
model = nengo.Network()
with model:
    if (type == "Unbalanced"):
        # Input Node - PresentInput can be used to present stimulus in time sequence
        stim = nengo.Node(PresentInput(mixture, presentation_time=presentation_time))

        ensemble = nengo.Ensemble(n_neurons=n_neurons, dimensions=dimensions, neuron_type=nengo.Izhikevich(tau_recovery=0.15))
        output = nengo.Node(size_in=dimensions)
        error = nengo.Node(size_in=dimensions)

        # Learning connections
        # conn = nengo.Connection(ensemble, output, function=lambda x: [0]*dimensions, 
        #                         learning_rule_type=nengo.PES(learning_rate))
        conn = nengo.Connection(ensemble, output, function=lambda x: [0]*dimensions)

        # Feedback for learning
        nengo.Connection(output, error)
        nengo.Connection(stim, error, transform=-1)
        # nengo.Connection(error, conn.learning_rule)

        # Input and Ensemble connections
        nengo.Connection(stim, ensemble)

        # Probe to monitor output
        output_probe = nengo.Probe(output, synapse=0.01)
        input_probe = nengo.Probe(stim, synapse=0.01)
        error_probe = nengo.Probe(error, synapse=0.01)

        stim_probe = nengo.Probe(stim, synapse=0.01)

        e_spk = nengo.Probe(ensemble.neurons) # spiking data from the ensemble

    elif (type == "Balanced"):
        stim = nengo.Node(PresentInput(mixture, presentation_time= presentation_time))

        # Ensembles
        exc_pop = nengo.Ensemble(n_neurons=num_exc, dimensions=dimensions, neuron_type=nengo.Izhikevich(tau_recovery=0.15))
        inh_pop = nengo.Ensemble(n_neurons=num_inh, dimensions=dimensions)

        null_pop = nengo.Ensemble(n_neurons = num_exc, dimensions = dimensions, neuron_type=nengo.Izhikevich(tau_recovery=0.15)) # neuron_type=nengo.Izhikevich(tau_recovery=0.1) # baseline to see whether signal is reproducible given population size

        output = nengo.Node(size_in=dimensions)
        error = nengo.Node(size_in=dimensions)
        
        ### NOTE: this currently excludes the balanced connectivity transformation from before ###
        nengo.Connection(stim, exc_pop) #input to excitatory
        nengo.Connection(exc_pop, inh_pop, transform=1) #excitatory to inhibitory
        nengo.Connection(inh_pop, exc_pop, transform=-1) #inhibitory to excitatory

        nengo.Connection(stim, null_pop)

        conn = nengo.Connection(exc_pop, output, function=lambda x: [0]*dimensions)
        nengo.Connection(output, error)
        nengo.Connection(stim, error, transform=-1)
        # nengo.Connection(error, conn.learning_rule)

        output_probe = nengo.Probe(output, synapse=0.01)
        input_probe = nengo.Probe(stim, synapse=0.01)
        error_probe = nengo.Probe(error, synapse=0.01)

        e_spk = nengo.Probe(exc_pop.neurons) # spiking data from the excitatory population
        i_spk = nengo.Probe(inh_pop.neurons) # spiking data from the inhibitory population
        n_spk = nengo.Probe(null_pop.neurons) # spiking data from the null population

# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(sim_time)
    t = sim.trange()

    #### NOTE: excitatory population metrics, pre-ablation ####
    exc_spk = sim.data[e_spk][:, :] # spiking at each milisecond -- (length of sim (ms) x num_neurons)
    pre_binary = exc_spk.T # (num_neurons x length of sim (ms))
    n_bin = int(len(pre_binary[0,:])/t_bin) # num bins required given bin time size
    pre_e_spike_count = np.zeros((len(pre_binary),n_bin)) # initialize spike count tracker -- (num_neurons x (ms / t_bin)) == (num_neurons x (n_bin))
    for j in range(n_bin - 1):
        pre_e_spike_count[:,j] = pre_binary[:,j*t_bin:(j+1)*t_bin].sum(axis=1) # populate spike counts given bin size
    X = pre_e_spike_count.transpose() # makes it a tall and skinny matrix
    X = StandardScaler().fit_transform(X) # scales the values to have zero mean and unit variance ()
    pre_cov_mat = np.cov(X, rowvar=False) # (nueron x neuron) covariance matrix based on inner product across all dimensions 
    pre_eig_vals, pre_eig_vect = np.linalg.eig(pre_cov_mat) # eigenvalues = (num_neurons x 1), eigenvectors = (num_neurons x num_neurons) -- each column is an eigenvector
    pre_eig_sum = np.sum(np.abs(pre_eig_vals)) # sum up all the eigenvalues
    povs_pre = np.sort(100*np.abs(pre_eig_vals)/pre_eig_sum)[::-1] # for later plotting

    #### NOTE: null population metrics, pre-ablation ####
    null_spk = sim.data[n_spk][:, :]
    pre_null_binary = null_spk.T
    pre_null_spike_count = np.zeros((len(pre_null_binary),n_bin))
    for j in range(n_bin - 1):
        pre_null_spike_count[:,j] = pre_null_binary[:,j*t_bin:(j+1)*t_bin].sum(axis=1)
    X_null = pre_null_spike_count.transpose()
    X_null = StandardScaler().fit_transform(X_null)
    pre_null_cov_mat = np.cov(X_null, rowvar=False)
    pre_null_eig_vals, pre_null_eig_vect = np.linalg.eig(pre_null_cov_mat)
    pre_null_eig_sum = np.sum(np.abs(pre_null_eig_vals))
    null_povs_pre = np.sort(100*np.abs(pre_null_eig_vals)/pre_null_eig_sum)[::-1]

    if (type == "Balanced"):
        ##### NOTE: joint population metrics, pre-ablation #####
        inh_spk = sim.data[i_spk][:, :]
        pre_i_binary = inh_spk.T
        pre_joint_binary = np.concatenate((pre_binary, pre_i_binary), axis=0)
        joint_n_bin = int(len(pre_joint_binary[0,:])/t_bin) # num bins required given bin time size
        pre_joint_spike_count = np.zeros((len(pre_joint_binary),joint_n_bin)) # initialize spike count tracker -- (num_neurons x (ms / t_bin)) == (num_neurons x (n_bin))
        for j in range(n_bin - 1):
            pre_joint_spike_count[:,j] = pre_joint_binary[:,j*t_bin:(j+1)*t_bin].sum(axis=1) # populate spike counts given bin size
        T = pre_joint_spike_count.transpose()
        T = StandardScaler().fit_transform(T)
        pre_joint_cov_mat = np.cov(T, rowvar=False)
        pre_joint_eig_vals, pre_joint_eig_vect = np.linalg.eig(pre_joint_cov_mat)
        pre_joint_eig_sum = np.sum(np.abs(pre_joint_eig_vals))
        joint_povs_pre = np.sort(100*np.abs(pre_joint_eig_vals)/pre_joint_eig_sum)[::-1]

    #--------------------------------------------------------------------------------------------------#
    # NOTE: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SVD analysis ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #--------------------------------------------------------------------------------------------------#    
    ##### NOTE: SVD power plots, post-ablation #####
    U, S, VT = np.linalg.svd(X, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    S_pre_ablation = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else

    ##### NOTE: SVD power plots, post-ablation, joint #####
    U, S, VT = np.linalg.svd(T, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    S_joint_pre_ablation = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else

    U, S, VT = np.linalg.svd(X_null, full_matrices=False)
    S_null_pre_ablation = np.diag(S)

    #--------------------------------------------------------------------------------------------------#

    pre_ablation_group_performance, pre_source_performance = average_classification_performance(X, source_activity)
    print(f"pre_ablation_group_performance: {pre_ablation_group_performance}")
    print(f"pre_source_performance: {pre_source_performance}")

    #--------------------------------------------------------------------------------------------------#

    if (type == "Unbalanced"):
        losers = ablate_population([ensemble], ablation_frac, sim)
    else:
        losers = ablate_population([exc_pop, inh_pop], ablation_frac, sim)

    ablate_population([null_pop], ablation_frac, sim)

    #--------------------------------------------------------------------------------------------------#
    
    sim.run(sim_time)

    #--------------------------------------------------------------------------------------------------#

    #### NOTE: excitatory population metrics, post-ablation ####
    exc_spk = sim.data[e_spk][((sim_time*1000)):, :] # spiking at each milisecond -- (length of sim (ms) x num_neurons)
    post_binary = exc_spk.T # (num_neurons x length of sim (ms))
    n_bin = int(len(post_binary[0,:])/t_bin) # num bins required given bin time size
    post_e_spike_count = np.zeros((len(post_binary),n_bin)) # initialize spike count tracker -- (num_neurons x (ms / t_bin)) == (num_neurons x (n_bin))
    for j in range(n_bin - 1):
        post_e_spike_count[:,j] = post_binary[:,j*t_bin:(j+1)*t_bin].sum(axis=1) # populate spike counts given bin size
    X_post = post_e_spike_count.transpose() # makes it a tall and skinny matrix
    X_post = StandardScaler().fit_transform(X_post) # scales the values to have zero mean and unit variance ()
    post_cov_mat = np.cov(X_post, rowvar=False) # (nueron x neuron) covariance matrix based on inner product across all dimensions 
    post_eig_vals, post_eig_vect = np.linalg.eig(post_cov_mat) # eigenvalues = (num_neurons x 1), eigenvectors = (num_neurons x num_neurons) -- each column is an eigenvector
    post_eig_sum = np.sum(np.abs(post_eig_vals)) # sum up all the eigenvalues
    povs_post = np.sort(100*np.abs(post_eig_vals)/post_eig_sum)[::-1] # save for later

    #### NOTE: null population metrics, post-ablation ####
    null_spk = sim.data[n_spk][((sim_time*1000)):, :]
    post_null_binary = null_spk.T
    post_null_spike_count = np.zeros((len(post_null_binary),n_bin))
    for j in range(n_bin - 1):
        post_null_spike_count[:,j] = post_null_binary[:,j*t_bin:(j+1)*t_bin].sum(axis=1)
    X_null_post = post_null_spike_count.transpose()
    X_null_post = StandardScaler().fit_transform(X_null_post)
    post_null_cov_mat = np.cov(X_null_post, rowvar=False)
    post_null_eig_vals, post_null_eig_vect = np.linalg.eig(post_null_cov_mat)
    post_null_eig_sum = np.sum(np.abs(post_null_eig_vals))
    null_povs_post = np.sort(100*np.abs(post_null_eig_vals)/post_null_eig_sum)[::-1]

    if (type == "Balanced"):
        ##### NOTE: joint population metrics, post-ablation #####
        inh_spk = sim.data[i_spk][sim_time*1000:, :]
        post_i_binary = inh_spk.T
        post_joint_binary = np.concatenate((post_binary, post_i_binary), axis=0)
        joint_n_bin = int(len(post_joint_binary[0,:])/t_bin) # num bins required given bin time size
        post_joint_spike_count = np.zeros((len(post_joint_binary),joint_n_bin)) # initialize spike count tracker -- (num_neurons x (ms / t_bin)) == (num_neurons x (n_bin))
        for j in range(n_bin - 1):
            post_joint_spike_count[:,j] = post_joint_binary[:,j*t_bin:(j+1)*t_bin].sum(axis=1) # populate spike counts given bin size
        T = post_joint_spike_count.transpose()
        T = StandardScaler().fit_transform(T)
        post_joint_cov_mat = np.cov(T, rowvar=False)
        post_joint_eig_vals, post_joint_eig_vect = np.linalg.eig(post_joint_cov_mat)
        post_joint_eig_sum = np.sum(np.abs(post_joint_eig_vals))
        joint_povs_post = np.sort(100*np.abs(post_joint_eig_vals)/post_joint_eig_sum)[::-1]

    #--------------------------------------------------------------------------------------------------#
    # NOTE: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SVD analysis ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #--------------------------------------------------------------------------------------------------#    
    ##### NOTE: SVD power plots, post-ablation #####
    U, S, VT = np.linalg.svd(X_post, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    S_post_ablation = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else

    ##### NOTE: SVD power plots, post-ablation, joint #####
    U, S, VT = np.linalg.svd(T, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    S_joint_post_ablation = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else

    U, S, VT = np.linalg.svd(X_null_post, full_matrices=False)
    S_null_post_ablation = np.diag(S)

    #--------------------------------------------------------------------------------------------------#
    # NOTE: ~~~~~~~~~~~~~~~~~~~~~~~~~~ post ablation classification ~~~~~~~~~~~~~~~~~~~~~~~~~
    #--------------------------------------------------------------------------------------------------#
        
    post_ablation_group_performance, post_source_performance = average_classification_performance(X_post, source_activity)
    print(f"post_ablation_group_performance: {post_ablation_group_performance}")
    print(f'post_source_performance: {post_source_performance}')

    #--------------------------------------------------------------------------------------------------#

    #-----------------------------------------------------------------------------------------------#
    ###### NOTE: state space plotting (pre-ablation) ######
    #--------------------------------------------------------------------------------------------------#
    exc_state_space = plt.figure() # Create a 3D plot
    ax = exc_state_space.add_subplot(111, projection='3d')
    ax.set_xlabel('PC 1', labelpad=-2) # Set labels for the axes
    ax.set_ylabel('PC 2', labelpad=-2)
    ax.set_zlabel('PC 3', labelpad=-2)
    ax.set_xticks([]) # remove the tick marks from the axis
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("State Space")
    cmap = plt.get_cmap('Blues')

    ##### NOTE: state space plotting, pre-ablation #####
    idx = pre_eig_vals.argsort()[::-1] # sort the eigenvalues and eigenvectors from highest to lowest  
    pre_eig_vals = pre_eig_vals[idx] # eigval shape is (40,)
    pre_eig_vect = pre_eig_vect[:,idx] # eigvector shape is (40 x 40) where each column represents an eigenvector
    eig3D = pre_eig_vect[:, :3]
    neural_state_space = pre_e_spike_count.T @ eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space
    num_time_steps = neural_state_space.shape[0] # Create a colormap based on the number of time steps
    print(f"num_time_steps: {num_time_steps}")

    color_weights = np.zeros((num_time_steps, 1)) # Create a list of colors for each time step

    for i in range(source_number):
        source_active = np.repeat(source_activity[i], (1000 / t_bin)) # scale this to match the dimensions of the state space plots
        for j in range(num_time_steps):
            if source_active[j] == 1:
                color_weights[j] += 1
    
    color_choices = ['yellow', 'orange', 'limegreen', 'navy']
    colors = []
    for color_weight in color_weights:
        colors.append(color_choices[int(color_weight - 1)])

    for i in range(num_time_steps): # plot the points with gradient coloring
        x = neural_state_space[i, 0]
        y = neural_state_space[i, 1]
        z = neural_state_space[i, 2]
        ax.scatter(x, y, z, c=colors[i], s=10)
        # ax.scatter(x, y, z, c='yellow' if np.repeat(half1_source, (1000 / t_bin))[i] == 0 else 'red', s=10, marker='D' if np.repeat(half2_source, (1000 / t_bin))[i] == 0 else 'x')

    ##### NOTE: plane plotting #####
    fit, residual = plane_of_best_fit(neural_state_space)
    xmin, xmax = np.min(neural_state_space[:, 0]), np.max(neural_state_space[:, 0]) # specify linspace overwhich to plot the plane based on datapoints
    ymin, ymax = np.min(neural_state_space[:, 1]), np.max(neural_state_space[:, 1])
    xx, yy = np.meshgrid(np.linspace(xmin, xmax), np.linspace(ymin, ymax)) # define the grid
    pre_z = (fit[0] * xx + fit[1] * yy + fit[2])
    pre_norm = (-fit[0], -fit[1], 1)
    ax.plot_surface(xx, yy, pre_z, alpha = 0.3, color = "lightblue") # plot the plane
    marker = line_integral(neural_state_space)
    ax.plot(neural_state_space[0:(num_time_steps - 1),0], neural_state_space[0:(num_time_steps - 1),1], neural_state_space[0:(num_time_steps - 1),2], c = "blue", label = f"pre-ablation trajectory", alpha = 0.4) # plot the trajectory

    ##### NOTE: post-ablation #####
    ### -- we're projecting the ablated data into the same eigenspace as the healthy data for comparison purposes -- ###
    neural_state_space = post_e_spike_count.T @ eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space
    cmap = plt.get_cmap('inferno')
    for i in range(num_time_steps - 1): # plot the points with gradient coloring
        x = neural_state_space[i, 0]
        y = neural_state_space[i, 1]
        z = neural_state_space[i, 2]
        ax.scatter(x, y, z, c=[cmap(i / num_time_steps)], s=10)  # Adjust s for point size
    marker = line_integral(neural_state_space)
    
    ##### NOTE: plane plotting #####
    fit, residual = plane_of_best_fit(neural_state_space)
    post_z = (fit[0] * xx + fit[1] * yy + fit[2])
    ax.plot_surface(xx, yy, post_z, alpha = 0.3, color = "lightcoral") # plot the plane
    post_norm = (-fit[0], -fit[1], 1)

    ##### NOTE: compute the difference in angle
    del_angle_radians = np.arccos(np.dot(pre_norm, post_norm) / (np.linalg.norm(pre_norm) * np.linalg.norm(post_norm)))
    del_angle_degrees = np.degrees(np.real(del_angle_radians))[0]

    ax.plot(neural_state_space[0:(num_time_steps - 1),0], neural_state_space[0:(num_time_steps - 1),1], neural_state_space[0:(num_time_steps - 1),2], c = "red", label = f"post-ablation trajectory", alpha = 0.4) # plot the trajectory

    ##### NOTE: making the state space plot look nice #####
    plt.legend()
    ax.view_init(elev=33, azim=11) # rotate the plot
    exc_state_space.savefig(path + "/state_space_rot_1", dpi = 300)
    ax.view_init(elev=46, azim=138) # rotate the plot
    exc_state_space.savefig(path + "/state_space_rot_2", dpi = 300)

    #-----------------------------------------------------------------------------------------------#

    ##### NOTE: pre ablation spike raster #####
    if (type == "Unbalanced"):
        fig, ax = plt.subplots()
        pre_binary = pre_binary[:, :time_cut_off]
        pre_spike_rates = np.sum(pre_binary, axis=1, keepdims=True) # sum across the time domain
        pre_spike_rate = np.sum(pre_spike_rates[:]) / (time_cut_off * n_neurons) # average across time
        for neuron_idx in range(n_neurons):
            spike_times = np.where(pre_binary[neuron_idx])[0]
            ax.scatter(spike_times, np.full_like(spike_times, neuron_idx), marker='|', color='green', s=0.1)
        ax.set_xlabel('Time (Seconds)')
        ax.set_ylabel('Neuron')
        tick_positions = np.arange(0, ax.get_xlim()[1] + 1, 5 * 1000)  # Calculate the tick positions
        ax.set_xticklabels(int(x) for x in ax.get_xticks() / 1000)  # Divide the x-axis tick labels by 1000
        ax.set_ylim(0, n_neurons)  # Set the y-axis range from 0 to 100
        ax.set_title('Pre-Ablation')
        plt.savefig(path + "/pre_ablation_spike_raster", dpi = 300)

        ##### NOTE: post ablation spike raster #####
        fig, ax = plt.subplots()
        post_binary = post_binary[:, :time_cut_off]
        post_spike_rates = np.sum(post_binary, axis=1, keepdims=True) # sum across the time domain
        post_spike_rate = np.sum(post_spike_rates[:]) / (time_cut_off * n_neurons) # average across time
        for neuron_idx in range(n_neurons):
            spike_times = np.where(post_binary[neuron_idx])[0]
            ax.scatter(spike_times, np.full_like(spike_times, neuron_idx), marker='|', color='green', s=0.5)
        ax.set_xlabel('Time (Seconds)')
        ax.set_ylabel('Neuron')
        ax.set_title('Post-Ablation')
        tick_positions = np.arange(0, ax.get_xlim()[1] + 1, 5 * 1000)  # Calculate the tick positions
        ax.set_xticks(tick_positions)  # Set the x-axis tick positions and labels
        ax.set_xticklabels([int(x) for x in tick_positions / 1000])  # Divide the x-axis tick labels by 1000
        ax.set_ylim(0, n_neurons)  # Set the y-axis range from 0 to 100
        plt.savefig(path + "/post_ablation_spike_raster", dpi = 300)
    
    elif (type == "Balanced"):
        ##### NOTE: pre-ablation raster plotting #####
        fig, ax = plt.subplots()
        pre_joint_binary = pre_joint_binary[:, :time_cut_off]
        pre_spike_rates = np.sum(pre_joint_binary, axis=1, keepdims=True) # sum across the time domain
        pre_exc_rate = np.sum(pre_spike_rates[:num_exc]) / (time_cut_off * num_exc) # average across time
        pre_inh_rate = np.sum(pre_spike_rates[num_exc : num_exc+num_inh]) / (time_cut_off * num_inh) # average across time
        for neuron_idx in range(num_exc + num_inh):
            spike_times = np.where(pre_joint_binary[neuron_idx])[0]
            ax.scatter(spike_times, np.full_like(spike_times, neuron_idx), marker='|', color='green' if neuron_idx < num_exc else 'red', s=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Neuron')
        ax.set_title('Pre-Ablation')
        ax.axhspan(0, num_exc, facecolor='green', alpha=0.05, label='excitatory neurons')
        ax.axhspan(num_exc, num_exc + num_inh, facecolor='red', alpha=0.05, label='inhibitory neurons')
        ax.set_ylim(-1, num_exc + num_inh)
        plt.savefig(path + "/pre_ablation_spike_raster", dpi = 300)

        ##### NOTE: post-ablation raster plotting #####
        fig, ax = plt.subplots()
        post_joint_binary = post_joint_binary[:, :time_cut_off]
        post_spike_rates = np.sum(post_joint_binary, axis=1, keepdims=True) # sum across the time domain
        post_exc_rate = np.sum(post_spike_rates[:num_exc]) / (time_cut_off * num_exc) # average across time
        post_inh_rate = np.sum(post_spike_rates[num_exc : num_exc+num_inh]) / (time_cut_off * num_inh) # average across time
        for neuron_idx in range(num_exc + num_inh):
            spike_times = np.where(post_joint_binary[neuron_idx])[0]
            ax.scatter(spike_times, np.full_like(spike_times, neuron_idx), marker='|', color='g' if neuron_idx < num_exc else 'red', s=0.5)
        if lesion_bars:
            for loser_y in losers: # show which neurons were ablated
                ax.axhline(y=loser_y, color='black', linewidth=1, alpha = 0.5) 
        ax.set_xlabel('Time')
        ax.set_ylabel('Neuron')
        ax.set_title('Post-Ablation')
        ax.axhspan(0, num_exc, facecolor='green', alpha=0.05, label='excitatory neurons')
        ax.axhspan(num_exc, num_exc + num_inh, facecolor='red', alpha=0.05, label='inhibitory neurons')
        ax.set_ylim(-1, num_exc + num_inh)
        plt.savefig(path + "/post_ablation_spike_raster", dpi = 300)

    #--------------------------------------------------------------------------------------------------#
    # NOTE: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ POV plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #--------------------------------------------------------------------------------------------------#    

    ##### NOTE: POV plotting, excitatory population only #####
    null_POV, null_var = plot_POVs([null_povs_pre, null_povs_post], [f"null, pre-ablation", f'null, post-ablation'], ["gold", "lightcoral"], f"null pop. variance explained by top {num_eigsum} PCs")
    null_POV.savefig(path + "/null_POV.png", dpi = 300)
    
    together_POV, together_var = plot_POVs([povs_pre, povs_post], [f"pre-ablation, {e_to_i*100}% connected", f"post-ablation, {e_to_i*100}% connected"], ["lightgreen", "lightblue"], f"excitatory pop. variance explained by top {num_eigsum} PCs")
    together_POV.savefig(path + "/together_POV.png", dpi = 300)

    ##### NOTE: POV plotting, entire population #####
    pop_POV, pop_var = plot_POVs([joint_povs_pre, joint_povs_post], [f"pre-ablation, {e_to_i*100}% connected", f"post-ablation, {e_to_i*100}% connected"], ["lightblue", "lightcoral"], f"whole pop. variance explained by top {num_eigsum} PCs")
    pop_POV.savefig(path + "/pop_POV.png", dpi = 300)

    ##### NOTE: SV/power plotting
    sv_plot = plt.figure() # singular value plotting
    plt.semilogy(np.diag(S_null_pre_ablation), linestyle = "dashdot", label="null pop, pre-ablation", color = "lightgreen")
    plt.semilogy(np.diag(S_null_post_ablation), linestyle = "dashdot", label="null pop, post-ablation", color = "darkgreen")
    plt.semilogy(np.diag(S_pre_ablation), linestyle = "dashed", label="exc pop, pre-ablation", color = "lightblue")
    plt.semilogy(np.diag(S_post_ablation), linestyle = "dashed", label="exc pop, post-ablation", color = "lightcoral")
    plt.semilogy(np.diag(S_joint_pre_ablation), linestyle = "solid", label="whole pop, pre-ablation", color = "lightblue")
    plt.semilogy(np.diag(S_joint_post_ablation), linestyle = "solid", label="whole pop, post-ablation", color = "lightcoral")
    plt.title('SV Log-Scale Power')
    plt.xlabel("singular value #")
    plt.ylabel("power of singular value")
    plt.legend()
    sv_plot.savefig(path + "/sv_plot.png", dpi = 300)

    sv_cum_contribution = plt.figure()
    plt.plot(np.cumsum((np.diag(S_null_pre_ablation))/np.sum(np.diag(S_null_pre_ablation)))*100, linestyle = "dashdot", label = "null pop, pre-ablation", color = "lightgreen")
    plt.plot(np.cumsum((np.diag(S_null_post_ablation))/np.sum(np.diag(S_null_post_ablation)))*100, linestyle = "dashdot", label = "null pop, post-ablation", color = "darkgreen")
    plt.plot(np.cumsum((np.diag(S_pre_ablation))/np.sum(np.diag(S_pre_ablation)))*100, linestyle = "dashed", label = "exc pop, pre-ablation", color = "lightblue")
    plt.plot(np.cumsum((np.diag(S_post_ablation))/np.sum(np.diag(S_post_ablation)))*100, linestyle = "dashed", label = "exc pop, post-ablation", color = "lightcoral")
    plt.plot(np.cumsum((np.diag(S_joint_pre_ablation))/np.sum(np.diag(S_joint_pre_ablation)))*100, linestyle = "solid", label = "whole pop, pre-ablation", color = "lightblue")
    plt.plot(np.cumsum((np.diag(S_joint_post_ablation))/np.sum(np.diag(S_joint_post_ablation)))*100, linestyle = "solid", label = "whole pop, post-ablation", color = "lightcoral")
    plt.title('SV Cumulative Contribution')
    plt.xlabel("# of singular values")
    plt.ylabel("percent contribution to overall data")
    plt.legend()
    sv_cum_contribution.savefig(path + "/sv_cum_contribution.png", dpi = 300)

    #--------------------------------------------------------------------------------------------------#
    # NOTE: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Performance Plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #--------------------------------------------------------------------------------------------------# 

    pre_accuracy = pre_ablation_group_performance['accuracy']
    post_accuracy = post_ablation_group_performance['accuracy']
    pre_post_plot = plt.figure()
    bar_width = 0.35
    indices = np.arange(source_number + 1)  # +1 for the cumulative bar

    # Colors for each source
    colors = plt.cm.viridis(np.linspace(0, 1, source_number))

    # Plotting bars for pre-ablation and post-ablation for each source
    for i in range(source_number):
        plt.bar(indices[i] - bar_width/2, pre_source_performance[i], color=colors[i], width=bar_width, label=f"Pre S{i+1}")
        plt.bar(indices[i] + bar_width/2, post_source_performance[i], color=colors[i], width=bar_width, label=f"Post S{i+1}")

    # Plotting the cumulative accuracy bars
    plt.bar(indices[-1] - bar_width/2, pre_accuracy, color='grey', width=bar_width, label='Pre Cumulative')
    plt.bar(indices[-1] + bar_width/2, post_accuracy, color='silver', width=bar_width, label='Post Cumulative')

    plt.xlabel('Source')
    plt.ylabel('Accuracy')
    plt.title('Pre and Post Ablation Accuracy by Source and Cumulatively')
    plt.xticks(indices - bar_width/2, [f"Source {i+1}" for i in range(source_number)] + ['Cumulative'])
    plt.legend()
    plt.savefig(path + "/pre_post_ablation_classification_performance.png", dpi=300)

fig, axs = plt.subplots(3, 3, figsize=(18,10))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure
# axs[0].imshow(plt.imread(path + "/half_1_pre_post_ablation_classification_performance.png"), aspect='auto')
# axs[0].imshow(plt.imread(path + "/pop_POV.png"), aspect='auto')
axs[1].imshow(plt.imread(path + "/state_space_rot_2.png"), aspect='auto')
axs[2].imshow(plt.imread(path + "/mixture_input.png"), aspect='auto')

# axs[3].imshow(plt.imread(path + "/half_2_pre_post_ablation_classification_performance.png"), aspect='auto')
# axs[3].imshow(plt.imread(path + "/together_POV.png"), aspect='auto')
axs[4].imshow(plt.imread(path + "/state_space_rot_1.png"), aspect='auto')
axs[5].imshow(plt.imread(path + "/pre_ablation_spike_raster.png"), aspect='auto')

# axs[6].imshow(plt.imread(path + "/sv_plot.png"), aspect='auto')
# axs[7].imshow(plt.imread(path + "/sv_cum_contribution.png"), aspect='auto')
axs[8].imshow(plt.imread(path + "/post_ablation_spike_raster.png"), aspect='auto')

fig.suptitle(f'{name}', fontsize = 10)
# fig.suptitle(f'{name}, pre-ablation -- MSE = {pre_mse:.3f}, CorrelationCoeff = {pre_correlation_coefficient:.3f}', fontsize = 10)
fig.savefig(path + "/pre_overview_fig.png", dpi = 300)
plt.tight_layout()

fig, axs = plt.subplots(3, 3, figsize=(18,10))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure
axs[0].imshow(plt.imread(path + "/null_POV.png"), aspect='auto')
axs[1].imshow(plt.imread(path + "/together_POV.png"), aspect='auto')
axs[2].imshow(plt.imread(path + "/pop_POV.png"), aspect='auto')

axs[3].imshow(plt.imread(path + "/sv_plot.png"), aspect='auto')
axs[4].imshow(plt.imread(path + "/sv_cum_contribution.png"), aspect='auto')
axs[5].imshow(plt.imread(path + "/pre_post_ablation_classification_performance.png"), aspect='auto')

# axs[6].imshow(plt.imread(path + "/null_POV.png"), aspect='auto')
# axs[7].imshow(plt.imread(path + "/pop_POV.png"), aspect='auto')
# axs[8].imshow(plt.imread(path + "/together_POV.png"), aspect='auto')
fig.suptitle(f"{name}", fontsize = 10)
fig.savefig(path + "/metrics_overview_fig.png", dpi = 300)
plt.tight_layout()

# ----------------------------------------------------------------------------------------------- #

##### NOTE: display/close functionality for plotting ease #####
plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run

# ----------------------------------------------------------------------------------------------- # 