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
OUT_DIR = "/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-01-26 Source Separation" #NOTE: change this for output folder
os.system('mkdir {}'.format(os.path.join(OUT_DIR,'plots')))

name = "Attempt 1 (dynamically caling number of sources in the isomura and furston model)"

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
        patch_half1_source = mpatches.Patch(color="yellow", label='Source 1 Used')
        patch_half2_source = mpatches.Patch(color="red", label='Source 2 Used')
        
        # Add the legend to the plot
        plt.legend(handles=[patch_half1, patch_half2, patch_half1_source, patch_half2_source], loc='upper right')

        plt.xlabel('Time Steps (Seconds)')
        plt.ylabel('Electrode')
        plt.title('Mixture Input Over Time')
        plt.xticks(np.arange(0, len(mixture_input)+1, 5), labels=np.arange(0, len(mixture_input)+1, 5))
        plt.savefig(path + "/mixture_input.png", dpi=300)

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

# ----------------------------------------------------------------------------------------------#

# generate the mixture input
trial_num = 30
trial_interval = 0 # the time between each trial (seconds)
trial_duration = 5 # the time that each trial lasts (seconds)
source_number = 20 # total number of sources
source_sample = 10 # how many sources we subsample from the total number of sources

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
dimensions = source_number
learning_rate = 1e-4
presentation_time = 1 # how long each stimulus is presented (seconds)
sim_time = trial_num * (trial_duration + trial_interval)

time_cut_off = (sim_time * 1000) # how many time steps to plot for the spike raster
testing_size = 0.3 # size of the testing set (fraction)

#---------------------------------------------------------------------------------------------#

mixture_input, half1_source, half2_source, half1, half2 = stochastic_mixture(trial_num=trial_num, trial_duration=trial_duration, 
                             trial_interval=trial_interval, source_number=source_number, 
                             source_sample=source_sample)

# np.set_printoptions(threshold=np.inf)
# print(f"mixture_input: {mixture_input}, with shape: {np.shape(mixture_input)}")
# np.set_printoptions(threshold=1000)
print(f"half 1 mixtrue input: {mixture_input[:,half1]}, with shape: {np.shape(mixture_input[:,half1])}")
print(f"half1_source: {half1_source}, with shape: {np.shape(half1_source)}")
print(f"half2_source: {half2_source}, with shape: {np.shape(half2_source)}")
print(f"half1: {half1}, with shape: {np.shape(half1)}")
print(f"half2: {half2}, with shape: {np.shape(half2)}")

# this is what we have
num_items = trial_num * (trial_duration + trial_interval) # this is the total number of time increments that are defined (number of samples)
num_features = len(half1) # this is the number of features that we have (number of electrodes corresponding to half 1)
# we're trying to differentiate between a zero (source 1) and one (source 2) at each time step

print(f"num_samples_time: {num_items}")
print(f"num_features: {num_features}")

# ----------------------------------------------------------------------------------------------#

plot_mixture_input_heatmap(mixture_input, half1, half2, half1_source, half2_source) # plot the mixture input

# ----------------------------------------------------------------------------------------------#

# Model network
model = nengo.Network()
with model:
    if (type == "Unbalanced"):
        # Input Node - PresentInput can be used to present stimulus in time sequence
        stim = nengo.Node(PresentInput(mixture_input, presentation_time=presentation_time))

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
        stim = nengo.Node(PresentInput(mixture_input, presentation_time= presentation_time))

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

    print(f"pre_binary: {pre_binary}, with shape: {np.shape(pre_binary)}")
    print(f"spike bins: {pre_e_spike_count.transpose()}, with shape: {np.shape(pre_e_spike_count.transpose())}")
    print(f"X: {X}, with shape: {np.shape(X)}") # (num_bins x num_neurons) -- 3999 x 100
    print(f" half1_source: {half2_source}, with shape: {np.shape(half2_source)}") # (num_bins x 1) -- 3999 x 1

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

    # matrix X has 100 features across 80 time points (num_neurons x num_bins) -- 100 * ____
    # half2_source has 1 feature across 80 time points (num_bins x 1)
        
    ##### NOTE: half 1 classification performance #####
    # splitting the dataset into training and testing sets
    X_train_pre_1, X_test_pre_1, y_train_pre_1, y_test_pre_1 = train_test_split(X, np.repeat(half1_source, (1000 / t_bin)), test_size=testing_size, random_state=42)
    log_reg_pre_1 = LogisticRegression()  # Initialize the Logistic Regression model
    log_reg_pre_1.fit(X_train_pre_1, y_train_pre_1)   # Train the model
    y_pred_pre_1 = log_reg_pre_1.predict(X_test_pre_1)  # Predicting the Test set results

    report_pre_1 = classification_report(y_test_pre_1, y_pred_pre_1, output_dict=True)   # Creating a classification report
    print(report_pre_1)

    print(report_pre_1['accuracy'])

    ##### NOTE: half 2 classification performance #####
    # splitting the dataset into training and testing sets
    X_train_pre_2, X_test_pre_2, y_train_pre_2, y_test_pre_2 = train_test_split(X, np.repeat(half2_source, (1000 / t_bin)), test_size=testing_size, random_state=42)
    log_reg_pre_2 = LogisticRegression()  # Initialize the Logistic Regression model
    log_reg_pre_2.fit(X_train_pre_2, y_train_pre_2)   # Train the model
    y_pred_pre_2 = log_reg_pre_2.predict(X_test_pre_2)  # Predicting the Test set results

    report_pre_2 = classification_report(y_test_pre_2, y_pred_pre_2, output_dict=True)   # Creating a classification report
    print(report_pre_2)

    # predictions: (This step would typically be applied to new, unseen data. Here, we use X_test for demonstration.)
    # predictions = log_reg.predict(X_test)
    # print(f"predictions {predictions}")

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

    # matrix X has 100 features across 80 time points (num_neurons x num_bins) -- 100 * ____
    # half2_source has 1 feature across 80 time points (num_bins x 1)
        
    ##### NOTE: half 1 classification performance #####
    # splitting the dataset into training and testing sets
    X_train_post_1, X_test_post_1, y_train_post_1, y_test_post_1 = train_test_split(X_post, np.repeat(half1_source, (1000 / t_bin)), test_size=testing_size, random_state=42)
    log_reg_post_1 = LogisticRegression()  # Initialize the Logistic Regression model
    log_reg_post_1.fit(X_train_post_1, y_train_post_1)   # Train the model
    y_pred_post_1 = log_reg_post_1.predict(X_test_post_1)  # Predicting the Test set results

    report_post_1 = classification_report(y_test_post_1, y_pred_post_1, output_dict=True)   # Creating a classification report
    print(report_post_1)

    ##### NOTE: half 2 classification performance #####
    # splitting the dataset into training and testing sets
    X_train_post_2, X_test_post_2, y_train_post_2, y_test_post_2 = train_test_split(X_post, np.repeat(half2_source, (1000 / t_bin)), test_size=testing_size, random_state=42)
    log_reg_post_2 = LogisticRegression()  # Initialize the Logistic Regression model
    log_reg_post_2.fit(X_train_post_2, y_train_post_2)   # Train the model
    y_pred_post_2 = log_reg_post_2.predict(X_test_post_2)  # Predicting the Test set results

    report_post_2 = classification_report(y_test_post_2, y_pred_post_2, output_dict=True)   # Creating a classification report
    print(report_post_2)

    #--------------------------------------------------------------------------------------------------#
    # NOTE: ~~~~~~~~~~~~~~~~~~~~~~~~~~ pre/post ablation classification performance ~~~~~~~~~~~~~~~~~~~~
    #--------------------------------------------------------------------------------------------------#

    df_pre = pd.DataFrame(report_pre_1).transpose()
    df_post = pd.DataFrame(report_post_1).transpose()

    pre_0 = df_pre.loc['0.0']   # Extracting data for '0.0' and '1.0'
    pre_1 = df_pre.loc['1.0']
    post_0 = df_post.loc['0.0']
    post_1 = df_post.loc['1.0']

    data = pd.DataFrame({  # Creating a DataFrame for easy plotting
        'Pre Source 1': pre_0,
        'Post Source 1': post_0,
        'Pre Source 2': pre_1,
        'Post Source 2': post_1
    })

    data = data.transpose() # Transpose the DataFrame for easier plotting
    data.drop('support', axis=1, inplace=True) # Drop the 'support' row

    ax = data.plot(kind='bar', figsize=(10, 6))

    plt.title('Half 1 Pre/Post Ablation Classification Performance')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.legend(title='Metrics')
    plt.tight_layout()

    for p in ax.patches:  # Add text entries with rounded values above each bar
        ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.savefig(path + "/half_1_pre_post_ablation_classification_performance.png", dpi=300)

    #--------------------------------------------------------------------------------------------------#

    df_pre = pd.DataFrame(report_pre_2).transpose()
    df_post = pd.DataFrame(report_post_2).transpose()

    pre_0 = df_pre.loc['0.0']   # Extracting data for '0.0' and '1.0'
    pre_1 = df_pre.loc['1.0']
    post_0 = df_post.loc['0.0']
    post_1 = df_post.loc['1.0']

    data = pd.DataFrame({  # Creating a DataFrame for easy plotting
        'Pre Source 1': pre_0,
        'Post Source 1': post_0,
        'Pre Source 2': pre_1,
        'Post Source 2': post_1
    })

    data = data.transpose() # Transpose the DataFrame for easier plotting
    data.drop('support', axis=1, inplace=True) # Drop the 'support' row

    ax = data.plot(kind='bar', figsize=(10, 6))

    plt.title('Half 2 Pre/Post Ablation Classification Performance')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.legend(title='Metrics')
    plt.tight_layout()

    for p in ax.patches:  # Add text entries with rounded values above each bar
        ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.savefig(path + "/half_2_pre_post_ablation_classification_performance.png", dpi=300)

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

    colors = []
    labels = []
    a = np.repeat(half1_source, (1000 / t_bin))
    b = np.repeat(half2_source, (1000 / t_bin))
    for i in range((num_time_steps)):
        if a[i] == 0 and b[i] == 0:
            colors.append('yellow')
            labels.append('H1S2 and H2S2')
        elif a[i] == 1 and b[i] == 0:
            colors.append('red')
            labels.append('H1S1 and H2S2')
        elif a[i] == 0 and b[i] == 1:
            colors.append('blue')
            labels.append('H1S2 and H2S1')
        else:
            colors.append('green')
            labels.append('H1S1 and H2S1')
    # Create a custom legend
    custom_legend = [plt.Line2D([], [], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
    plt.legend(custom_legend, labels, title='Point Colors')

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

fig, axs = plt.subplots(3, 3, figsize=(18,10))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure
axs[0].imshow(plt.imread(path + "/half_1_pre_post_ablation_classification_performance.png"), aspect='auto')
axs[1].imshow(plt.imread(path + "/state_space_rot_2.png"), aspect='auto')
axs[2].imshow(plt.imread(path + "/mixture_input.png"), aspect='auto')

axs[3].imshow(plt.imread(path + "/half_2_pre_post_ablation_classification_performance.png"), aspect='auto')
axs[4].imshow(plt.imread(path + "/state_space_rot_1.png"), aspect='auto')
axs[5].imshow(plt.imread(path + "/pre_ablation_spike_raster.png"), aspect='auto')

# axs[6].imshow(plt.imread(path + "/state_space_rot_1.png"), aspect='auto')
# axs[7].imshow(plt.imread(path + "/state_space_rot_2.png"), aspect='auto')
axs[8].imshow(plt.imread(path + "/post_ablation_spike_raster.png"), aspect='auto')

fig.suptitle(f'{name}', fontsize = 10)
# fig.suptitle(f'{name}, pre-ablation -- MSE = {pre_mse:.3f}, CorrelationCoeff = {pre_correlation_coefficient:.3f}', fontsize = 10)
fig.savefig(path + "/pre_overview_fig", dpi = 300)
plt.tight_layout()

# ----------------------------------------------------------------------------------------------- #

##### NOTE: display/close functionality for plotting ease #####
plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run

# ----------------------------------------------------------------------------------------------- # 