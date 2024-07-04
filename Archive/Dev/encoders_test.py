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

# generate the mixture input
trial_num = 30 # number of trials NOTE changed from 50
trial_interval = 0 # the time between each trial (seconds)
trial_duration = 5 # the time that each trial lasts (seconds)
electrode_number = 32 #TODO 64 # NOTE total number of stimulation sites
source_number = 2 #TODO 4 # NOTE number of inputs we're trying to separate from
presentation_time = 1 # how long each stimulus is presented (seconds)

# model parameters
n_neurons = 10 # number of neurons in the ensemble
t_bin = 100 # size of spike count bins (ms)
dt = 0.001 # time step (ms), so each point is a ms

type = "Unbalanced" # "Unbalanced" or "Balanced"
# e_to_i = 0.7 # excitatory to inhibitory ratio
ablation_frac = 0.90 # fraction of neurons to ablate
num_iter = 21 # number of iterations

# misc setup
e_to_i = 0.7
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

mixture, source_activity, input_stim_sites = stochastic_mixture(trial_num=trial_num, trial_duration=trial_duration, 
                                    trial_interval=trial_interval, electrode_number=electrode_number, num_sources=source_number,
                                    electrode_sample=electrode_sample)

# Model network
model = nengo.Network()
with nengo.Network() as model:
    if (type == "Unbalanced"):
        # Input Node - PresentInput can be used to present stimulus in time sequence
        stim = nengo.Node(PresentInput(mixture, presentation_time=presentation_time))

        ensemble = nengo.Ensemble(n_neurons=n_neurons, dimensions=dimensions, neuron_type=nengo.Izhikevich(tau_recovery=0.15))
        output = nengo.Node(size_in=dimensions)
        error = nengo.Node(size_in=dimensions)

        # Learning connections
        # conn = nengo.Connection(ensemble, output, function=lambda x: [0]*dimensions, 
        #                         learning_rule_type=nengo.PES(learning_rate))
        conn = nengo.Connection(ensemble, output, function=lambda x: [0]*dimensions, solver=nengo.solvers.LstsqL2(weights=True))

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

# with nengo.Simulator(model) as simbuild:

    # weights = simbuild.data[ensemble].encoders

    # built_ensemble = simbuild.data[ensemble]


    # # Get the "x" values (evaluation points scaled by encoders) for ensemble
    # x_vals = np.dot(built_ensemble.eval_points, built_ensemble.encoders.T / ensemble.radius)

    # # Get the activity values corresponding to the x values 
    # activities = ensemble.neuron_type.rates(x_vals, built_ensemble.gain, built_ensemble.bias)

    # # Create the solver, and use it to solve for the decoders of ensemble that compute a specific output function
    # solver = nengo.solvers.LstsqL2(weights=True)
    # decoders, _ = solver(activities, built_ensemble.eval_points)
    # # # Note that if the ensemble is doing a communication channel, then no `output_func` call is needed:
    # # decoders, _ = solver(activities, built_ensemble.eval_points)

    # # Compute the full weight matrix by doing the matrix multiplication with the encoders of ens2
    # weights = np.dot(simbuild.data[ens2].encoders, decoders.T)

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

    weights = sim.data[ensemble].encoders

    # weights = sim.signals[sim.model.sig[conn]["weights"]]


    print(f"weights  = {weights}")
    print(f"weights shape = {weights.shape}")

    print(f"first weight = {weights[0]}")

    print(f"1023rd weight = {weights[1023]}")

    # print(f"exc_spk = {exc_spk}")
    # print(f"exc_spk shape = {exc_spk.shape}")