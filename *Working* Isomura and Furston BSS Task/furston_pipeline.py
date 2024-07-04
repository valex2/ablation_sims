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
import csv

# os.environ
# OUT_DIR = "/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-01-26 Source Separation" #NOTE: change this for output folder
# os.system('mkdir {}'.format(os.path.join(OUT_DIR,'plots')))

# name = "Attempt 1 (dynamically caling number of sources in the isomura and furston model)"

# path = os.path.join(OUT_DIR,f'{name} plots')
# if not os.path.exists(path):
#         os.makedirs(path)

##### NOTE: CREATE SIMULATION FOLDER AND CONFIG #####
os.environ
RUNNER = os.environ['RUNNER']
ARRAY_ID = int(os.environ['NUM_JOB']) # ARRAY ID CAN BE USED TO INDEX INTO A VARIABLE OF INTEREST --> param = param_range[ARRAY_ID]

# source_range = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8] # number of sources we're trying to seperate from (32 entries)
source_range = [1, 2, 3, 4] # number of sources we're trying to seperate from (32 entries)
# electrode_number = [32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256]
electrode_number = 16 # TODO: change this to vary across electrodes

# source_number = source_range[ARRAY_ID]
# electrode_number = electrode_number[ARRAY_ID]

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
trial_num = 30 # number of trials
trial_interval = 0 # the time between each trial (seconds)
trial_duration = 5 # the time that each trial lasts (seconds)
# electrode_number = 64 # total number of stimulation sites
# source_number = 3 # number of inputs we're trying to seperate from

# model parameters
n_neurons = 100 # number of neurons in the ensemble
t_bin = 100 # size of spike count bins (ms)
dt=0.001 # time step (ms), so each point is a ms

type = "Balanced" # "Unbalanced" or "Balanced"
lesion_bars = False # True or False
e_to_i = 0.7
num_exc = int(n_neurons * e_to_i)
num_inh = n_neurons - num_exc

ablation_frac = 1 # fraction of neurons to ablate
dimensions = electrode_number
presentation_time = 1 # how long each stimulus is presented (seconds)
sim_time = trial_num * (trial_duration + trial_interval)

time_cut_off = (sim_time * 1000) # how many time steps to plot for the spike raster
testing_size = 0.3 # size of the testing set (fraction)

### PARMATERS YAML ###
runID_num = '01'
slurm_dict = {
    'trial_num': trial_num,
    'trial_interval': trial_interval,
    'trial_duration': trial_duration,
    'electrode_number': electrode_number,
    # 'electrode_sample': electrode_sample,
    # 'source_number': source_number,

    'n_neurons': n_neurons,
    't_bin': t_bin,
    'dt': dt,

    'type': type,
    'lesion_bars': lesion_bars,
    'e_to_i': e_to_i,
    'num_exc': num_exc,
    'num_inh': num_inh,

    'ablation_frac': ablation_frac,
    'dimensions': dimensions,
    'presentation_time': presentation_time,
    'sim_time': sim_time,

    'time_cut_off': time_cut_off,
    'testing_size': testing_size
}

with open(os.path.join(path,'model_slurm_config.yml'), 'w+') as cfg_file:    # add out here and to all other scripts, remove config move call below
    yaml.dump(slurm_dict,cfg_file,default_flow_style=False)

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
        }
    }

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

    # Averaging the performance metrics
    for key in total_performance.keys():
        if isinstance(total_performance[key], dict):
            for metric in total_performance[key].keys():
                total_performance[key][metric] /= num_sources

    return total_performance

def main():
    source_performance_pre = {}
    source_performance_post = {}
    for source_number in source_range:
        print(f"source_number: {source_number}")
        electrode_sample = int(np.floor(electrode_number/source_number))# how many stimulation sites are associated with each source (can overlap)

        for iter in range(1): # this can be used to average across trials with the same hyperparameter set
            print(f" iteration: {iter}")

            mixture, source_activity, input_stim_sites = stochastic_mixture(trial_num=trial_num, trial_duration=trial_duration, 
                                        trial_interval=trial_interval, electrode_number=electrode_number, num_sources=source_number,
                                        electrode_sample=electrode_sample)

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

                    output = nengo.Node(size_in=dimensions)
                    error = nengo.Node(size_in=dimensions)
                    
                    ### NOTE: this currently excludes the balanced connectivity transformation from before ###
                    nengo.Connection(stim, exc_pop) #input to excitatory
                    nengo.Connection(exc_pop, inh_pop, transform=1) #excitatory to inhibitory
                    nengo.Connection(inh_pop, exc_pop, transform=-1) #inhibitory to excitatory

                    conn = nengo.Connection(exc_pop, output, function=lambda x: [0]*dimensions)
                    nengo.Connection(output, error)
                    nengo.Connection(stim, error, transform=-1)
                    # nengo.Connection(error, conn.learning_rule)

                    output_probe = nengo.Probe(output, synapse=0.01)
                    input_probe = nengo.Probe(stim, synapse=0.01)
                    error_probe = nengo.Probe(error, synapse=0.01)

                    e_spk = nengo.Probe(exc_pop.neurons) # spiking data from the excitatory population
                    i_spk = nengo.Probe(inh_pop.neurons) # spiking data from the inhibitory population

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

                pre_ablation_group_performance = average_classification_performance(X, source_activity)
                print(f"pre_ablation_group_performance: {pre_ablation_group_performance}")
                source_performance_pre[source_number] = pre_ablation_group_performance

                #--------------------------------------------------------------------------------------------------#

                if (type == "Unbalanced"):
                    losers = ablate_population([ensemble], ablation_frac, sim)
                else:
                    losers = ablate_population([exc_pop, inh_pop], ablation_frac, sim)

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
                # NOTE: ~~~~~~~~~~~~~~~~~~~~~~~~~~ post ablation classification ~~~~~~~~~~~~~~~~~~~~~~~~~
                #--------------------------------------------------------------------------------------------------#
                    
                post_ablation_group_performance = average_classification_performance(X_post, source_activity)
                print(f"post_ablation_group_performance: {post_ablation_group_performance}")
                source_performance_post[source_number] = post_ablation_group_performance

    #--------------------------------------------------------------------------------------------------#
    # NOTE: ~~~~~~~~~~~~~~~~~~~~~~~~~~ saving data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #--------------------------------------------------------------------------------------------------#

    experiment_path = os.path.join(path, f'furston_classification_pre_electrodes={electrode_number}')  # Main directory
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    data = [] # deal with the pre-ablation data
    for source, performance in source_performance_pre.items():
        for status, metrics in performance.items():
            metrics['status'] = status
            metrics['source'] = source
            data.append(metrics)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Split the DataFrame based on the status
    df_off = df[df['status'] == 'off']
    df_on = df[df['status'] == 'on']

    # file path pre-ablation
    file_path_off = os.path.join(experiment_path,f'performance_off_pre_{ARRAY_ID}.csv')
    file_path_on = os.path.join(experiment_path,f'performance_on_pre_{ARRAY_ID}.csv')

    # Save to separate CSV files
    df_off.to_csv(file_path_off, index=False)
    df_on.to_csv(file_path_on, index=False)

    experiment_path = os.path.join(path, f'furston_classification_post_electrodes={electrode_number}')  # Main directory
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    data = [] # deal with the pre-ablation data
    for source, performance in source_performance_post.items():
        for status, metrics in performance.items():
            metrics['status'] = status
            metrics['source'] = source
            data.append(metrics)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Split the DataFrame based on the status
    df_off = df[df['status'] == 'off']
    df_on = df[df['status'] == 'on']

    # file path pre-ablation
    file_path_off = os.path.join(experiment_path, f'performance_off_post_{ARRAY_ID}.csv')
    file_path_on = os.path.join(experiment_path, f'performance_on_post_{ARRAY_ID}.csv')

    # Save to separate CSV files
    df_off.to_csv(file_path_off, index=False)
    df_on.to_csv(file_path_on, index=False)

if __name__ == '__main__':
  main()