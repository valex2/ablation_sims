# import math
import os
# import random
# import sys
# import matplotlib as mpl
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
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

"""The goal of this file is to create a rapidly iterable framework for 
evaluating the distance between two covariance matrices that are computed 
after trials of a simulation. This is a data generation stage of a larger pipeline.
The pipeline is as follows:
1. Perform 40 simulations of a model with a given set of parameters (sweeping over the E:I ratio)
2. Compute the covariance matrices for the excitatory population before and after ablation
3. Compute pairwise distances between the pre and post ablation covariance matrices for each of those 40 simulations
4. Save the distances to a file --> create a 3D parameter spcae plot of the distances
"""

##### NOTE: CREATE SIMULATION FOLDER AND CONFIG #####
os.environ
RUNNER = os.environ['RUNNER']
ARRAY_ID = int(os.environ['NUM_JOB']) # ARRAY ID CAN BE USED TO INDEX INTO A VARIABLE OF INTEREST --> param = param_range[ARRAY_ID]

e_to_i_range = np.arange(0.1, 0.91, 0.01) # 81 different parameter values
e_to_i = e_to_i_range[ARRAY_ID]

OUT_DIR = os.path.join(os.environ['DM_T'])
vassilis_out_dir = os.path.join(OUT_DIR,'vassilis_out')
if os.path.exists(vassilis_out_dir):
    print('vassilis_out folder present')
    #sys.exit(1)
else:
    # os.system('mkdir {}'.format(os.path.join(OUT_DIR,'vassilis_out')))
    os.makedirs(vassilis_out_dir)
path = os.path.join(OUT_DIR,'vassilis_out')

# generate the mixture input
trial_num = 30 # number of trials NOTE changed from 50
trial_interval = 0 # the time between each trial (seconds)
trial_duration = 5 # the time that each trial lasts (seconds)
electrode_number = 64 #TODO 64 # NOTE total number of stimulation sites
source_number = 4 #TODO 4 # NOTE number of inputs we're trying to separate from
presentation_time = 1 # how long each stimulus is presented (seconds)

folder_title = '4x64_weighted_eigs'

# model parameters
n_neurons = 100 # number of neurons in the ensemble
t_bin = 100 # size of spike count bins (ms)
dt = 0.001 # time step (ms), so each point is a ms

type = "Weighted" # "Unbalanced" or "Balanced"
# e_to_i = 0.7 # excitatory to inhibitory ratio
ablation_frac = 0.90 # fraction of neurons to ablate
num_iter = 21 # number of iterations

# misc setup
num_exc = int(n_neurons * e_to_i)
num_inh = n_neurons - num_exc
dimensions = electrode_number
lesion_bars = False # True or False
if type == "Continuous":
    sim_time = 30
else:
    sim_time = trial_num * (trial_duration + trial_interval)
electrode_sample = int(np.floor(electrode_number / source_number)) # how many stimulation sites are associated with each source (can overlap)

e_i_precent_conn = 0.05
i_e_precent_conn = 0.05

time_cut_off = (sim_time * 1000) # how many time steps to plot for the spike raster
testing_size = 0.3 # size of the testing set (fraction)
num_eigsum = 15 # number of eigenvalues to sum for the POV
learning_rate = 1e-4

### PARAMETERS YAML ###
runID_num = '01'
slurm_dict = {
    'runID_num': runID_num,
    'trial_num': trial_num,
    'trial_interval': trial_interval,
    'trial_duration': trial_duration,
    'electrode_number': electrode_number,
    'source_number': source_number,
    'presentation_time': presentation_time,

    'n_neurons': n_neurons,
    't_bin': t_bin,
    'dt': dt,

    'type': type,
    'e_to_i': e_to_i,
    'ablation_frac': ablation_frac,

    'num_exc': num_exc,
    'num_inh': num_inh,
    'dimensions': dimensions,
    'lesion_bars': lesion_bars,
    'sim_time': sim_time,
    'electrode_sample': electrode_sample,

    'e_i_precent_conn': e_i_precent_conn,
    'i_e_precent_conn': i_e_precent_conn,

    'time_cut_off': time_cut_off,
    'testing_size': testing_size,
    'num_eigsum': num_eigsum,
    'learning_rate': learning_rate
}

with open(os.path.join(path,'model_slurm_config.yml'), 'w+') as cfg_file:    # add out here and to all other scripts, remove config move call below
    yaml.dump(slurm_dict,cfg_file,default_flow_style=False)

print(f"\n\n\n THIS IS CHILD {ARRAY_ID} with E:I ratio {e_to_i:2f} -- {num_exc}:{num_inh} -- out of {e_to_i_range} \n\n\n")

##### NOTE: this function generates a layered sinusoid #####
def layered_periodic_function(t):
    # the sum of all of these is still 2pi periodic
    cos_frequencies = np.array([1]) * np.pi
    sin_frequencies = np.array([1, 2, 6]) * np.pi

    cos_terms = np.cos(np.outer(cos_frequencies, t)) # plug in t
    sin_terms = np.sin(np.outer(sin_frequencies, t))

    cos_sum = np.sum(cos_terms, axis=0) # sum across the 0 axis
    sin_sum = np.sum(sin_terms, axis=0)
    
    return (cos_sum + sin_sum) / (len(cos_frequencies) + len(sin_frequencies)) # standardize by the number of things being summed

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
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
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
        denominator = exc_sums[j] + abs(inh_sums[j])
        if denominator == 0:
            abs_scaling_factor = 0  # or another appropriate default value
        else:
            abs_scaling_factor = abs(inh_sums[j]) / denominator        
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

def main():
    pre_matrices = [] # store the pre-ablation covariance matrices
    post_matrices = [] # store the post-ablation covariance matrices
    diff_matrices = [] # store the difference between pre and post ablation covariance matrices

    pre_joint_matrices = [] # store the pre-ablation joint covariance matrices
    post_joint_matrices = [] # store the post-ablation joint covariance matrices
    diff_joint_matrices = [] # store the difference between pre and post ablation joint covariance matrices

    pre_accuracy_vect = [] # store the pre-ablation classification accuracy
    post_accuracy_vect = [] # store the post-ablation classification accuracy
    diff_accuracy_vect = [] # store the difference between pre and post ablation classification accuracy

    joint_pre_accuracy_vect = [] # store the pre-ablation joint classification accuracy
    joint_post_accuracy_vect = [] # store the post-ablation joint classification accuracy
    joint_diff_accuracy_vect = [] # store the difference between pre and post ablation joint classification accuracy

    pre_exc_encoder_matrices = [] # store the excitatory encoder matrices
    pre_inh_encoder_matrices = [] # store the inhibitory encoder matrices

    post_exc_encoder_matrices = [] # store the excitatory encoder matrices
    post_inh_encoder_matrices = [] # store the inhibitory encoder matrices

    pre_eigvals1 = [] # store the pre-ablation eigenvalues
    pre_eigvals2 = []
    pre_eigvals3 = []

    pre_joint_eigvals1 = [] # store the pre-ablation joint eigenvalues
    pre_joint_eigvals2 = []
    pre_joint_eigvals3 = []

    post_eigvals1 = [] # store the post-ablation eigenvalues
    post_eigvals2 = []
    post_eigvals3 = []

    post_joint_eigvals1 = [] # store the post-ablation joint eigenvalues
    post_joint_eigvals2 = []
    post_joint_eigvals3 = []

    for iter in range(num_iter):
        print(f"ITERATION {iter}")

        ###### -------------------------------------------------------------------------------------------------- ######

        mixture, source_activity, input_stim_sites = stochastic_mixture(trial_num=trial_num, trial_duration=trial_duration, 
                                    trial_interval=trial_interval, electrode_number=electrode_number, num_sources=source_number,
                                    electrode_sample=electrode_sample)

        ###### -------------------------------------------------------------------------------------------------- ######

        # Model network
        model = nengo.Network()
        with model:
            if (type == "Continuous"):
                # Input Node - continuous
                stim = nengo.Node(output=layered_periodic_function)

                exc_pop = nengo.Ensemble(n_neurons=num_exc, dimensions= 1, neuron_type=nengo.Izhikevich(tau_recovery=0.15))
                inh_pop = nengo.Ensemble(n_neurons=num_inh, dimensions= 1)
                null_pop = nengo.Ensemble(n_neurons = num_exc, dimensions = 1, neuron_type=nengo.Izhikevich(tau_recovery=0.15)) # neuron_type=nengo.Izhikevich(tau_recovery=0.1) # baseline to see whether signal is reproducible given population size


                conn, e_e, e_i, i_e, i_i = balance_condition(num_exc, num_inh, e_i_precent_conn, i_e_precent_conn)

                nengo.Connection(stim, exc_pop) #input to excitatory
                nengo.Connection(exc_pop.neurons, exc_pop.neurons, transform = e_e)
                nengo.Connection(exc_pop.neurons, inh_pop.neurons, transform = e_i) # network connections
                nengo.Connection(inh_pop.neurons, exc_pop.neurons, transform = i_e)
                nengo.Connection(inh_pop.neurons, inh_pop.neurons, transform = i_i)

                nengo.Connection(stim, null_pop)

                e_spk = nengo.Probe(exc_pop.neurons) # spiking data from the excitatory population
                i_spk = nengo.Probe(inh_pop.neurons) # spiking data from the inhibitory population
                n_spk = nengo.Probe(null_pop.neurons) # spiking data from the null population

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

            elif (type == "Weighted"):
                stim = nengo.Node(PresentInput(mixture, presentation_time= presentation_time))

                exc_pop = nengo.Ensemble(n_neurons=num_exc, dimensions=dimensions, neuron_type=nengo.Izhikevich(tau_recovery=0.15))
                inh_pop = nengo.Ensemble(n_neurons=num_inh, dimensions=dimensions)
                null_pop = nengo.Ensemble(n_neurons = num_exc, dimensions = dimensions, neuron_type=nengo.Izhikevich(tau_recovery=0.15)) # neuron_type=nengo.Izhikevich(tau_recovery=0.1) # baseline to see whether signal is reproducible given population size


                conn, e_e, e_i, i_e, i_i = balance_condition(num_exc, num_inh, e_i_precent_conn, i_e_precent_conn)

                nengo.Connection(stim, exc_pop) #input to excitatory
                nengo.Connection(exc_pop.neurons, exc_pop.neurons, transform = e_e)
                nengo.Connection(exc_pop.neurons, inh_pop.neurons, transform = e_i) # network connections
                nengo.Connection(inh_pop.neurons, exc_pop.neurons, transform = i_e)
                nengo.Connection(inh_pop.neurons, inh_pop.neurons, transform = i_i)

                nengo.Connection(stim, null_pop)


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

            if (type == "Balanced" or type == "Weighted" or type == "Continuous"):
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

            # now I want to store the top three eigenvalues for each iteration
            sorted_pre_eigvals = np.sort(np.abs(pre_eig_vals))
            sorted_pre_eigvals = sorted_pre_eigvals[::-1]
            pre_eigvals1.append(sorted_pre_eigvals[0])
            pre_eigvals2.append(sorted_pre_eigvals[1])
            pre_eigvals3.append(sorted_pre_eigvals[2])

            sorted_pre_joint_eigvals = np.sort(np.abs(pre_joint_eig_vals))
            sorted_pre_joint_eigvals = sorted_pre_joint_eigvals[::-1]
            pre_joint_eigvals1.append(sorted_pre_joint_eigvals[0])
            pre_joint_eigvals2.append(sorted_pre_joint_eigvals[1])
            pre_joint_eigvals3.append(sorted_pre_joint_eigvals[2])

            if type != "Continuous":
                pre_ablation_group_performance, pre_source_performance = average_classification_performance(X, source_activity)
                joint_pre_ablation_group_performance, joint_pre_source_performance = average_classification_performance(T, source_activity)

            pre_exc_encoder = sim.data[exc_pop].encoders
            pre_inh_encoder = sim.data[inh_pop].encoders

            pre_exc_encoder_matrices.append(pre_exc_encoder)
            pre_inh_encoder_matrices.append(pre_inh_encoder)

        #--------------------------------------------------------------------------------------------------#

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

            if (type == "Balanced" or type == "Weighted" or type == "Continuous"):
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


            # again, grab the eigenvales
            sorted_post_eigvals = np.sort(np.abs(post_eig_vals))
            sorted_post_eigvals = sorted_post_eigvals[::-1]
            post_eigvals1.append(sorted_post_eigvals[0])
            post_eigvals2.append(sorted_post_eigvals[1])
            post_eigvals3.append(sorted_post_eigvals[2])

            sorted_post_joint_eigvals = np.sort(np.abs(post_joint_eig_vals))
            sorted_post_joint_eigvals = sorted_post_joint_eigvals[::-1]
            post_joint_eigvals1.append(sorted_post_joint_eigvals[0])
            post_joint_eigvals2.append(sorted_post_joint_eigvals[1])
            post_joint_eigvals3.append(sorted_post_joint_eigvals[2])

            if type != "Continuous":
                post_ablation_group_performance, post_source_performance = average_classification_performance(X_post, source_activity) 
                joint_post_ablation_group_performance, joint_post_source_performance = average_classification_performance(T, source_activity)

            post_exc_encoder = sim.data[exc_pop].encoders
            post_inh_encoder = sim.data[inh_pop].encoders
            post_exc_encoder_matrices.append(post_exc_encoder)
            post_inh_encoder_matrices.append(post_inh_encoder)

            #--------------------------------------------------------------------------------------------------#
            # distance saving
            pre_matrices.append(pre_cov_mat)
            post_matrices.append(post_cov_mat)
            diff_matrices.append(pre_cov_mat - post_cov_mat)

            pre_joint_matrices.append(pre_joint_cov_mat)
            post_joint_matrices.append(post_joint_cov_mat)
            diff_joint_matrices.append(pre_joint_cov_mat - post_joint_cov_mat)

            #--------------------------------------------------------------------------------------------------#
            # accuracy handling
            if type != "Continuous":
                pre_accuracy = pre_ablation_group_performance['accuracy'] # pull the entire accuracy metrics
                post_accuracy = post_ablation_group_performance['accuracy']
                diff_accuracy = pre_accuracy - post_accuracy

                pre_accuracy_vect.append(pre_accuracy)
                post_accuracy_vect.append(post_accuracy)
                diff_accuracy_vect.append(diff_accuracy)

                joint_pre_accuracy = joint_pre_ablation_group_performance['accuracy'] # pull the entire accuracy metrics
                joint_post_accuracy = joint_post_ablation_group_performance['accuracy']
                joint_diff_accuracy = joint_pre_accuracy - joint_post_accuracy

                joint_pre_accuracy_vect.append(joint_pre_accuracy)
                joint_post_accuracy_vect.append(joint_post_accuracy)
                joint_diff_accuracy_vect.append(joint_diff_accuracy)
            
            #--------------------------------------------------------------------------------------------------#
            # print(frobenius_distance(pre_cov_mat, post_cov_mat))
            # print(frobenius_distance(pre_cov_mat, pre_cov_mat))
            #--------------------------------------------------------------------------------------------------#

    #--------------------------------------------------------------------------------------------------#
    # compute pairwise distances + save
    pre_distances = compute_pairwise_distances(pre_matrices)
    post_distances = compute_pairwise_distances(post_matrices)   
    diff_distances = compute_pairwise_distances(diff_matrices)

    pre_joint_distances = compute_pairwise_distances(pre_joint_matrices)
    post_joint_distances = compute_pairwise_distances(post_joint_matrices)
    diff_joint_distances = compute_pairwise_distances(diff_joint_matrices)

    pre_exc_distances = compute_pairwise_distances(pre_exc_encoder_matrices)
    pre_inh_distances = compute_pairwise_distances(pre_inh_encoder_matrices)

    post_exc_distances = compute_pairwise_distances(post_exc_encoder_matrices)
    post_inh_distances = compute_pairwise_distances(post_inh_encoder_matrices)

    #--------------------------------------------------------------------------------------------------#
    # NOTE: ~~~~~~~~~~~~~~~~~~~~~~~~~~ saving data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #--------------------------------------------------------------------------------------------------#
    # save the distances
    experiment_path = os.path.join(path, f'{folder_title}')  # Main directory
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    dist_path = os.path.join(experiment_path, f'distances')  # Main directory
    if not os.path.exists(dist_path):
        os.makedirs(dist_path)

    file_path = os.path.join(dist_path, f'{ARRAY_ID}.csv')
    
        
    df_dist = pd.DataFrame({'pre': pre_distances, 'post': post_distances, 'diff': diff_distances, 
                       'pre_joint': pre_joint_distances, 'post_joint': post_joint_distances, 'diff_joint': diff_joint_distances, 
                       'pre_exc_enc': pre_exc_distances, 'pre_inh_enc': pre_inh_distances, 'post_exc_enc': post_exc_distances, 'post_inh_enc': post_inh_distances})
    
    if type != "Continuous":
        df_acc = pd.DataFrame({'pre': pre_accuracy_vect, 'post': post_accuracy_vect, 'diff': diff_accuracy_vect, 
                            'joint_pre': joint_pre_accuracy_vect, 'joint_post': joint_post_accuracy_vect, 'joint_diff': joint_diff_accuracy_vect})
    
    df_eigs = pd.DataFrame({'pre_eig1': pre_eigvals1, 'pre_eig2': pre_eigvals2, 'pre_eig3': pre_eigvals3, 
                            'pre_joint_eig1': pre_joint_eigvals1, 'pre_joint_eig2': pre_joint_eigvals2, 'pre_joint_eig3': pre_joint_eigvals3,
                            'post_eig1': post_eigvals1, 'post_eig2': post_eigvals2, 'post_eig3': post_eigvals3, 
                            'post_joint_eig1': post_joint_eigvals1, 'post_joint_eig2': post_joint_eigvals2, 'post_joint_eig3': post_joint_eigvals3})

    # Save the DataFrame to a CSV file
    df_dist.to_csv(file_path, index=False)

    print(f'saved to file {file_path}')

    if type != "Continuous":
        acc_path = os.path.join(experiment_path, f'accuracy')  # Main directory
        if not os.path.exists(acc_path):
            os.makedirs(acc_path)

        file_path = os.path.join(acc_path, f'{ARRAY_ID}.csv')

        df_acc.to_csv(file_path, index=False)

        print(f'saved to file {file_path}')

    eigs_path = os.path.join(experiment_path, f'eigenvalues')  # Main directory
    if not os.path.exists(eigs_path):
        os.makedirs(eigs_path)

    file_path = os.path.join(eigs_path, f'{ARRAY_ID}.csv')

    df_eigs.to_csv(file_path, index=False)

    print(f'saved to file {file_path}')

if __name__ == '__main__':
    main()