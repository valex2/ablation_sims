# import math
import os
# import random
# import sys
# import matplotlib as mpl
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nengo
# from nengo.processes import WhiteSignal
from nengo.processes import PresentInput
import pandas as pd
# import scipy.signal as sig
# import scipy.stats as stats
import yaml
# from tqdm import tqdm
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap
# from matplotlib.lines import Line2D
# from nengo.processes import Piecewise, WhiteSignal
# from nengo.utils.matplotlib import rasterplot
# from scipy.optimize import curve_fit
# from sklearn.decomposition import PCA
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Line3DCollection
# from sklearn.linear_model import LinearRegression
# from matplotlib.gridspec import GridSpec
# import networkx as nx
# import seaborn as sns
# from scipy.interpolate import interp1d
# from scipy.linalg import lstsq
# from scipy.integrate import quad
import numpy as np
# from nengo.solvers import LstsqL2
# import nengo_bio as bio
from sklearn.model_selection import train_test_split
# from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

"""The goal of this file is to see how training with different
excitatory to inhibitory ratios (i.e. different population sizes)
affects scalings of the encoder matrices
We want to see scalings that are not linearly dependent on size
"""

# generate the mixture input
trial_num = 30 # number of trials NOTE changed from 50
trial_interval = 0 # the time between each trial (seconds)
trial_duration = 5 # the time that each trial lasts (seconds)
electrode_number = 64 #TODO 64 # NOTE total number of stimulation sites
source_number = 4 #TODO 4 # NOTE number of inputs we're trying to separate from
presentation_time = 1 # how long each stimulus is presented (seconds)

# model parameters
n_neurons = 100 # number of neurons in the ensemble
t_bin = 100 # size of spike count bins (ms)
dt = 0.001 # time step (ms), so each point is a ms

type = "Balanced" # "Unbalanced" or "Balanced"
# e_to_i = 0.7 # excitatory to inhibitory ratio
ablation_frac = 0.90 # fraction of neurons to ablate
num_iter = 21 # number of iterations

# misc setup
e_to_i = 0.4 # TODO scale this to see how it affects encoder matrices
num_exc = int(n_neurons * e_to_i)
num_inh = n_neurons - num_exc
dimensions = electrode_number
lesion_bars = False # True or False
sim_time = trial_num * (trial_duration + trial_interval)
electrode_sample = int(np.floor(electrode_number / source_number)) # how many stimulation sites are associated with each source (can overlap)

e_i_precent_conn = 0.05
i_e_precent_conn = 0.05

time_cut_off = (sim_time * 1000) # how many time steps to plot for the spike raster
testing_size = 0.3 # size of the testing set (fraction)
num_eigsum = 15 # number of eigenvalues to sum for the POV
learning_rate = 1e-4

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

# NOTE: this function computes the frobenius distance between two matrices
# inputs: two matrices
# outputs: the frobenius distance between the two matrices (||A - B||_F)
def frobenius_distance(A, B):
    return np.linalg.norm(A - B, 'fro')

# NOTE: this function computes the pairwise distances between matrices using the Frobenius norm
def compute_pairwise_distances(matrices):
    n = len(matrices)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            # Compute the Frobenius norm of the difference between matrices i and j
            distance = frobenius_distance(matrices[i], matrices[j])
            distances.append(distance)
    return distances

# NOTE: given a precent connectivity, this function outputs the corresponding transform matrix
# inputs: num_exc_neur -- number of excitatory neurons
# num_inh_neur -- number of inhibitory neurons
# e_i_precent_conn -- measure of connectivity between excitatory and inhibitory populations
# i_e_precent_conn -- measure of connectivity between inhibitory and excitatory populations
# output: connectivity_matrix, excitatory_transform_e_e, excitatory_transform_e_i, inhibitory_transform_i_e, inhibitory_transform_i_i
def balance_condition(num_exc_neur, num_inh_neur, e_i_precent_conn, i_e_precent_conn):
    connectivity_matrix = np.zeros(((num_exc_neur + num_inh_neur), (num_exc_neur + num_inh_neur)))

    #### NOTE: excitatory connections ####
    desired_e_i_connections = ((num_exc_neur * num_inh_neur)*e_i_precent_conn)
    connected = 0 # tracking the num of connections formed in transform matrix
    while connected < desired_e_i_connections:
        row = np.random.randint(0, num_exc_neur)
        col = np.random.randint(num_exc_neur, num_exc_neur + num_inh_neur)
         # Check if the connection is not already established
        if connectivity_matrix[row, col] == 0:
            connectivity_matrix[row, col] = np.random.normal(0.1, 0.05)  # Set the connection
            connected += 1  # Increment the counter

    #### NOTE: inhibitory connections ####
    desired_i_e_connections = ((num_exc_neur * num_inh_neur)*i_e_precent_conn)
    connected = 0 # tracking the num of connections formed in transform matrix
    while connected < desired_i_e_connections:
        row = np.random.randint(num_exc_neur, num_exc_neur + num_inh_neur)
        col = np.random.randint(0, num_exc_neur)
         # Check if the connection is not already established
        if connectivity_matrix[row, col] == 0:
            connectivity_matrix[row, col] = np.random.normal(-0.1, 0.05)  # Set the connection
            connected += 1  # Increment the counter

    col_sums = np.sum(connectivity_matrix, axis=0)
    #### NOTE: excitatory recurrent connections ####
    exc_sum = col_sums[:num_exc_neur]
    num_e_e_conns = np.ceil(np.abs(exc_sum))
    for i, conns in enumerate(num_e_e_conns):
        connected = 0
        while connected < conns:
            row = np.random.randint(0, num_exc_neur)
            col = i # fix the column
            if connectivity_matrix[row, col] == 0:
                connectivity_matrix[row, col] = np.random.normal(3, 0.5)  # Set the connection
                connected += 1  # Increment the counter

    #### NOTE: inhibitory recurrent connections ####
    inh_sum = col_sums[num_exc_neur:]
    num_i_i_conns = np.ceil(np.abs(inh_sum))
    for i, conns in enumerate(num_i_i_conns):
        connected = 0
        while connected < conns:
            row = np.random.randint(num_exc_neur, num_exc_neur + num_inh_neur)
            col = num_exc_neur + i # fix the column
            if connectivity_matrix[row, col] == 0:
                connectivity_matrix[row, col] = np.random.normal(-3, 0.5)  # Set the connection
                connected += 1  # Increment the counter

    #### NOTE: normalization ####
    exc_sums = np.sum(connectivity_matrix[:num_exc_neur, :], axis=0)
    inh_sums = np.sum(connectivity_matrix[num_exc_neur:, :], axis=0)

    weights_matrix = np.zeros((2, num_exc_neur + num_inh_neur))
    for j in range(num_exc_neur + num_inh_neur):
        # Calculate the absolute value of the scaling factor
        abs_scaling_factor = abs(inh_sums[j]) / (exc_sums[j] + abs(inh_sums[j]))
        
        # Assign the scaling factors to the weights matrix
        weights_matrix[0, j] = abs_scaling_factor
        weights_matrix[1, j] = 1 - abs_scaling_factor  # Ensuring the sum of weights is 1

    ##### NOTE: applying weights #####
    for i in range(len(connectivity_matrix[0, :])):
        connectivity_matrix[:num_exc_neur, i] = connectivity_matrix[:num_exc_neur, i] * weights_matrix[0, i]
        connectivity_matrix[num_exc_neur:, i] = connectivity_matrix[num_exc_neur:, i] * weights_matrix[1, i]

    connectivity_matrix = np.nan_to_num(connectivity_matrix).T

    ##### NOTE: pull out sub-matrices #####
    excitatory_transform_e_e = connectivity_matrix[:num_exc_neur, :num_exc_neur]
    inhibitory_transform_i_e = connectivity_matrix[:num_exc_neur, num_exc_neur:]
    excitatory_transform_e_i = connectivity_matrix[num_exc_neur:, :num_exc_neur]
    inhibitory_transform_i_i = connectivity_matrix[num_exc_neur:, num_exc_neur:]

    return connectivity_matrix, excitatory_transform_e_e, excitatory_transform_e_i, inhibitory_transform_i_e, inhibitory_transform_i_i

on = False

if on:
    # vector to store the 3d vectors
    povs_3D_all = []

    # e_to_i = 0.4 # TODO scale this to see how it affects encoder matrices
    for e_to_i in np.linspace(0.1, 0.9, 10):

        num_exc = int(n_neurons * e_to_i)
        num_inh = n_neurons - num_exc

        mixture, source_activity, input_stim_sites = stochastic_mixture(trial_num=trial_num, trial_duration=trial_duration, 
                                        trial_interval=trial_interval, electrode_number=electrode_number, num_sources=source_number,
                                        electrode_sample=electrode_sample)

        model = nengo.Network()
        with model:
            stim = nengo.Node(PresentInput(mixture, presentation_time= presentation_time))

            # Ensembles
            exc_pop = nengo.Ensemble(n_neurons=num_exc, dimensions=dimensions, neuron_type=nengo.Izhikevich(tau_recovery=0.15))
            inh_pop = nengo.Ensemble(n_neurons=num_inh, dimensions=dimensions)

            null_pop = nengo.Ensemble(n_neurons = num_exc, dimensions = dimensions, neuron_type=nengo.Izhikevich(tau_recovery=0.15)) # neuron_type=nengo.Izhikevich(tau_recovery=0.1) # baseline to see whether signal is reproducible given population size

            output = nengo.Node(size_in=dimensions)
            error = nengo.Node(size_in=dimensions)
            
            ### NOTE: this currently excludes the balanced connectivity transformation from before ###
            conn_in = nengo.Connection(stim, exc_pop) #input to excitatory
            conn_e_i = nengo.Connection(exc_pop, inh_pop, transform=1) #excitatory to inhibitory
            conn_i_e = nengo.Connection(inh_pop, exc_pop, transform=-1) #inhibitory to excitatory

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

            scaled_encoders = nengo.Probe(exc_pop, 'scaled_encoders')

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
            # print the weights
            # weights = sim.data[exc_pop].encoders
            # print(f"weights  = {weights}")
            # print(f"weights shape = {weights.shape}")
            # print(f"first weight = {weights[0]}")
            # print(f"1023rd weight = {weights[1023]}")
            # print the encoders
            encoders = sim.data[scaled_encoders][1,:,:]
            # print(f"encoders = {encoders}")
            # print(f"encoders shape = {encoders.shape}")
            encoders_scaled = StandardScaler().fit_transform(encoders)
            # print(f"encoders_scaled = {encoders_scaled}")
            encoders_cov_mat = np.cov(encoders_scaled, rowvar=False)
            # print(f"encoders_cov_mat = {encoders_cov_mat}")
            # print(f"encoders_cov_mat shape = {encoders_cov_mat.shape}")
            encoders_eig_vals, encoders_eig_vect = np.linalg.eig(encoders_cov_mat)
            # print(f"encoders_eig_vals = {encoders_eig_vals}")
            # print(f"encoder_eig_vals shape = {encoders_eig_vals.shape}")
            eig_sum = np.sum(np.abs(encoders_eig_vals))
            # print(f"eig_sum = {eig_sum}")
            povs = np.sort(100*np.abs(encoders_eig_vals)/eig_sum)[::-1]
            # print(f"povs size = {povs.shape}")
            povs_3D = povs[:3] 
            povs_3D = np.real(povs_3D)
            # print(f"povs_3D = {povs_3D}")
            povs_3D_all.append(povs_3D)

            # save the povs_3D_all vector locally
            np.save("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/povs_3D_all.npy", povs_3D_all)

# plot each of the 3d vectors in 3 space with a different color for each e_to_i ratio
# povs_3D_all = np.array(povs_3D_all)


povs_3D_all = np.load("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/povs_3D_all.npy")

labels = np.array([0.1, 0.18888889, 0.27777778, 0.36666667, 0.45555556,
                   0.54444444, 0.63333333, 0.72222222, 0.81111111, 0.9])

# Create the plot
figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')

# Plot each point in a different color based on its label
colors = plt.cm.viridis((labels - labels.min()) / (labels.max() - labels.min()))

for i in range(len(povs_3D_all)):
    ax.scatter(povs_3D_all[i,0], povs_3D_all[i,1], povs_3D_all[i,2], color=colors[i], label=f'{labels[i]:.2f}')

# Setting labels for the axes
ax.set_xlabel('1st POV')
ax.set_ylabel('2nd POV')
ax.set_zlabel('3rd POV')

# Creating a custom legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = list(sorted(set(labels), key=labels.index))
unique_handles = [handles[labels.index(l)] for l in unique_labels]
ax.legend(unique_handles, unique_labels, title="E:I Ratio", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title("Excitatory Population Encoder Scaling")
plt.savefig("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/encoder_scaling.png", dpi = 300)

plt.show()

# print(conn_in.probeable)
# print(exc_pop.probeable)
# print(stim.probeable)

