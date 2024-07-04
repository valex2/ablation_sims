import math
import os
import csv
import random
import sys
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nengo
from nengo.processes import WhiteSignal
import numpy as np
import pandas as pd
import scipy.signal as sig
import scipy.stats as stats
import scipy.spatial as spatial
import yaml
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

def balance_condition(num_exc_neur, num_inh_neur, e_i_precent_conn, i_e_precent_conn):
    connectivity_matrix = np.zeros(((num_exc_neur + num_inh_neur), (num_exc_neur + num_inh_neur)))
    e_i_ratio = num_exc_neur/num_inh_neur

    #### NOTE: excitatory connections ####
    desired_e_i_connections = ((num_exc_neur * num_inh_neur)*e_i_precent_conn)
    connected = 0 # tracking the num of connections formed in transform matrix
    while connected < desired_e_i_connections:
        row = np.random.randint(0, num_exc_neur)
        col = np.random.randint(num_exc_neur, num_exc_neur + num_inh_neur)
         # Check if the connection is not already established
        if connectivity_matrix[row, col] == 0:
            connectivity_matrix[row, col] = np.random.gumbel(0.1, 0.05) #np.random.normal(0.1, 0.05)  # Set the connection
            connected += 1  # Increment the counter

    #### NOTE: inhibitory connections ####
    desired_i_e_connections = ((num_exc_neur * num_inh_neur)*i_e_precent_conn)
    connected = 0 # tracking the num of connections formed in transform matrix
    while connected < desired_i_e_connections:
        row = np.random.randint(num_exc_neur, num_exc_neur + num_inh_neur)
        col = np.random.randint(0, num_exc_neur)
         # Check if the connection is not already established
        if connectivity_matrix[row, col] == 0:
            connectivity_matrix[row, col] =  np.random.gumbel(-0.1, 0.05) #np.random.normal(-0.1, 0.05)  # Set the connection
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
                connectivity_matrix[row, col] =  np.random.gumbel(3, 0.5)#np.random.normal(3, 0.5)  # Set the connection
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
                connectivity_matrix[row, col] =  np.random.gumbel(-3, 0.5)#np.random.normal(-3, 0.5)  # Set the connection
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

def ablate_population(ensembles, proportion, sim, bias=True):
    # NOTE: population-level ablation (has the capability to lesion across neuronal ensembles)
    # inputs: vector of ensembles we want to lesion, proportion of overall neurons that should be lesioned
    # outputs: none, but affects n neurons in each ensemble such that their encoder values are 0 and bias terms are -1000
    # -- ensembles of form [exc_pop, inh_pop, .........]
    # -- proportion: fraction of neurons that we want to ablate
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
                    bias_sig[rel_neur_idx] = -1000000 # the bias term is set to 0, making the populations un-excitable
            
        group_idx = group_idx + ens.n_neurons
    return losers

def layered_periodic_function(t):
    ##### NOTE: this function generates a layered sinusoid #####
    # the sum of all of these is still 2pi periodic
    cos_frequencies = np.array([1]) * np.pi
    sin_frequencies = np.array([1, 2, 6]) * np.pi

    cos_terms = np.cos(np.outer(cos_frequencies, t)) # plug in t
    sin_terms = np.sin(np.outer(sin_frequencies, t))

    cos_sum = np.sum(cos_terms, axis=0) # sum across the 0 axis
    sin_sum = np.sum(sin_terms, axis=0)
    
    return (cos_sum + sin_sum) / (len(cos_frequencies) + len(sin_frequencies)) # standardize by the number of things being summed

def calculate_mse(baseline, for_comparison):
    #### NOTE: this function computes the mean square error between two sets of data (the data must have the same length)
    if len(for_comparison) != len(baseline):
        raise ValueError("Signals must have the same length")
    
    squared_diff = np.square(baseline - for_comparison)
    mse = np.mean(squared_diff)
    return mse

model = nengo.Network(label="E/I offset")

##### META-GOAL -- for each E:I ratio vary across ablation frac and compare MSE offsets for both null and connected groups ####

##### NOTE: CREATE SIMULATION FOLDER AND CONFIG #####
os.environ
RUNNER = os.environ['RUNNER']
ARRAY_ID = int(os.environ['NUM_JOB']) # ARRAY ID CAN BE USED TO INDEX INTO A VARIABLE OF INTEREST --> param = param_range[ARRAY_ID]

#e_i_ratio_range = [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2] # fraction of neurons that are excitatory -- looping for each ratio -- requesting 8 (avoiding 0.1 to avoid same-norm systems)
e_i_ratio_range = np.arange(0.9, 0.2 -0.025, -0.025) # 29 steps
# ablation_frac_range = np.linspace(0.00, 1, 25)
ablation_frac = 0.7 # setting constant ablation at 0.5
rel_e_i = e_i_ratio_range[ARRAY_ID] # pull this based off of the run ID
NUM_TOTAL = 100 # number of total neurons for rel e:i
num_exc = int(np.ceil(NUM_TOTAL * rel_e_i))
num_inh = int(np.ceil(NUM_TOTAL * (1 - rel_e_i)))

##### NOTE: CREATE SIMULATION FOLDER AND CONFIG #####
OUT_DIR = os.path.join(os.environ['DM_T'])
vassilis_out_dir = os.path.join(OUT_DIR,'vassilis_out')
if os.path.exists(vassilis_out_dir):
    print('vassilis_out folder present')
    #sys.exit(1)
else:
    # os.system('mkdir {}'.format(os.path.join(OUT_DIR,'vassilis_out')))
    os.makedirs(vassilis_out_dir)
path = os.path.join(OUT_DIR,'vassilis_out')

print(f"\n\n\n THIS IS CHILD {ARRAY_ID} with E:I ratio {rel_e_i:2f} -- {num_exc}:{num_inh} -- out of {e_i_ratio_range} \n\n\n")

### PARMATERS YAML ###
runID_num = '01'
slurm_dict =  {
              'num_exc_neur' : num_exc, # 360
              'num_inh_neur' : num_inh, # 40
              #'ablation_frac' : 0.5, # fraction of neurons to lesion
              
              'e_i_precent_conn' : 0.05, # excitatory connectivity ratio
              'e_i_max_val' : 0.05, # the greatest strength of the e_i connection
              'e_i_tailed_dist': True, # whether excitatory connection strengths are drawn from a tailed distribution (with max val) or not

              'i_e_precent_conn' : 0.05, # inhibitory connectivity ratio
              'i_e_max_val' : -0.1, # the greatest strength of the i_e connection
              'i_e_tailed_dist': True, # whether inhibitory connection strengths are drawn from a tailed distribution (with max val) or not
              
              'i_i_precent_conn' : 1, # recurrent inhibitory connectivity ratio
              'i_i_max_val' : -1, # the greatest strength of the i_i connection
              'i_i_tailed_dist': True, # whether recurrent connection strengths are drawn from a tailed distribution (with max val) or not

              'lesion_e' : True, # whether or not to lesion the excitatory population
              'lesion_i' : True, # whether or not to lesion the inhibitory population

              'probe_synapse' : 0.05, # synaptic filter on the network outputs (seconds)
              'sim_duration' : 4, # (seconds)
              'dt' : 0.001, # # plot time increment
              't_bin' : 20, # size of binning spike counts (ms)

              'num_eigsum' : 10, # number of PCs we plot POV for
              'lesion_bars' : False, # whether the lesion horizontal bars should be plotted or not
              'time_cut_off': 4000, # how many data points we plot for spike rasters
}

with open(os.path.join(path,'model_slurm_config.yml'), 'w+') as cfg_file:    # add out here and to all other scripts, remove config move call below
    yaml.dump(slurm_dict,cfg_file,default_flow_style=False)

num_exc_neur = slurm_dict['num_exc_neur']
num_inh_neur = slurm_dict['num_inh_neur']
lesion_e = slurm_dict['lesion_e']
lesion_i = slurm_dict['lesion_i']

e_i_precent_conn = slurm_dict['e_i_precent_conn']
e_i_tailed_dist = slurm_dict['e_i_tailed_dist']
e_i_max_val = slurm_dict['e_i_max_val']

i_e_precent_conn = slurm_dict['i_e_precent_conn']
i_e_tailed_dist = slurm_dict['i_e_tailed_dist']
i_e_max_val = slurm_dict['i_e_max_val']

i_i_precent_conn = slurm_dict['i_i_precent_conn']
i_i_tailed_dist = slurm_dict['i_i_tailed_dist']
i_i_max_val = slurm_dict['i_i_max_val']

num_eigsum = slurm_dict['num_eigsum']
probe_synapse = slurm_dict['probe_synapse']
sim_duration = slurm_dict['sim_duration']
dt = slurm_dict['dt']
t_bin = slurm_dict['t_bin']
# ablation_frac = slurm_dict['ablation_frac']
time_cut_off = slurm_dict['time_cut_off']
lesion_bars = slurm_dict['lesion_bars']

pre_spaces = [] # store the spaces
post_spaces = [] # store the spaces
for iter in range(25):
    print(f"iter num {iter}")
    with model: 
        ### population initialization ###o
        rand_num = np.random.randint(0, 4000) # seed the null and exc population such that they're the same
        exc_pop = nengo.Ensemble(n_neurons = num_exc_neur, dimensions = 1, seed = rand_num)
        inh_pop = nengo.Ensemble(n_neurons = num_inh_neur, dimensions = 1)
        null_pop = nengo.Ensemble(n_neurons = num_exc_neur, dimensions = 1, seed = rand_num) # baseline to see whether signal is reproducible given population size

        input_signal = nengo.Node(output=layered_periodic_function)

        # NOTE: a transform is a linear transformation  mapping the pre function output to the post function input
        conn, e_e, e_i, i_e, i_i = balance_condition(num_exc_neur, num_inh_neur, e_i_precent_conn, i_e_precent_conn)

        nengo.Connection(input_signal, exc_pop) #input to excitatory
        nengo.Connection(exc_pop.neurons, exc_pop.neurons, transform = e_e)
        nengo.Connection(exc_pop.neurons, inh_pop.neurons, transform = e_i) # network connections
        nengo.Connection(inh_pop.neurons, exc_pop.neurons, transform = i_e)
        nengo.Connection(inh_pop.neurons, inh_pop.neurons, transform = i_i)
        nengo.Connection(input_signal, null_pop) # connect to the null_pop
        
        ### probing ###
        e_probe = nengo.Probe(exc_pop, synapse=probe_synapse)
        null_probe = nengo.Probe(null_pop, synapse=probe_synapse)

        e_spikes = nengo.Probe(exc_pop.neurons)

    with nengo.Simulator(model) as sim:
        ##### NOTE: pre-ablation run #####
        sim.run(sim_duration)
        t = sim.trange()
        pre_ablation_e_probe = sim.data[e_probe][10:, :] # value at each milisecond

        ### NOTE: binning spike counts -- pre-ablation ###
        e_spk = sim.data[e_spikes][10:, :] # spiking at each milisecond -- (length of sim (ms) x num_neurons)
        pre_binary = e_spk.T # (num_neurons x length of sim (ms))
        n_bin = int(len(pre_binary[0,:])/t_bin) # num bins required given bin time size
        pre_e_spike_count = np.zeros((len(pre_binary),n_bin)) # initialize spike count tracker -- (num_neurons x (ms / t_bin)) == (num_neurons x (n_bin))
        for j in range(n_bin - 1):
            pre_e_spike_count[:,j] = pre_binary[:,j*t_bin:(j+1)*t_bin].sum(axis=1) # populate spike counts given bin size
        X = pre_e_spike_count.transpose() # makes it a tall and skinny matrix
        X = StandardScaler().fit_transform(X) # scales the values to have zero mean and unit variance ()
        pre_cov_mat = np.cov(X, rowvar=False) # (nueron x neuron) covariance matrix based on inner product across all dimensions 
        pre_eig_vals, pre_eig_vect = np.linalg.eig(pre_cov_mat)

        ### project into 3D space ###
        idx = pre_eig_vals.argsort()[::-1] # sort the eigenvalues and eigenvectors from highest to lowest  
        pre_eig_vals = pre_eig_vals[idx] # eigval shape is (40,)
        pre_eig_vect = pre_eig_vect[:,idx] # eigvector shape is (40 x 40) where each column represents an eigenvector
        eig3D = pre_eig_vect[:, :3]
        pre_neural_state_space = pre_e_spike_count.T @ eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space
        pre_spaces.append(pre_neural_state_space)

        #------------------------------------------------------------------------------------------------------------------------#
    
        losers = ablate_population([exc_pop, inh_pop], ablation_frac, sim)
        ablate_population([null_pop], ablation_frac, sim)

        ##### NOTE: post-ablation run #####
        sim.run(sim_duration)
        t = sim.trange()
        post_ablation_e_probe = sim.data[e_probe][sim_duration*1000 + 10:, :] # value at each milisecond
        post_ablation_null_probe = sim.data[null_probe][sim_duration*1000 + 10:, :]

        ### NOTE: binning spike counts -- post-ablation ###
        e_spk = sim.data[e_spikes][sim_duration*1000 + 10:, :] # spiking at each milisecond -- (length of sim (ms) x num_neurons)
        post_ablation_e_probe = sim.data[e_probe][sim_duration*1000 + 10:, :] # value at each milisecond
        post_ablation_null_probe = sim.data[null_probe][sim_duration*1000 + 10:, :]
        post_binary = e_spk.T # (num_neurons x length of sim (ms))
        n_bin = int(len(post_binary[0,:])/t_bin) # num bins required given bin time size
        post_e_spike_count = np.zeros((len(post_binary),n_bin)) # initialize spike count tracker -- (num_neurons x (ms / t_bin)) == (num_neurons x (n_bin))
        for j in range(n_bin - 1):
            post_e_spike_count[:,j] = post_binary[:,j*t_bin:(j+1)*t_bin].sum(axis=1) # populate spike counts given bin size
        X = post_e_spike_count.transpose() # makes it a tall and skinny matrix
        X = StandardScaler().fit_transform(X) # scales the values to have zero mean and unit variance ()
        post_cov_mat = np.cov(X, rowvar=False) # (nueron x neuron) covariance matrix based on inner product across all dimensions 
        post_eig_vals, post_eig_vect = np.linalg.eig(post_cov_mat) # eigenvalues = (num_neurons x 1), eigenvectors = (num_neurons x num_neurons) -- each column is an eigenvector

        post_neural_state_space = post_e_spike_count.T @ eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space
        post_spaces.append(post_neural_state_space)

def pairwise_procrustes_similarity(spaces):
    """
    Computes the summed procrustes similarity between all iterations of a given model (ground truth)
    Returns the average similarity across all iterations
    """
    similarities = []
    for space in spaces:
        for space2 in spaces:
            if space is not space2:
                procrustes_similarity = spatial.procrustes(space, space2)[2]
                similarities.append(procrustes_similarity)
    return np.average(similarities), np.std(similarities), np.std(similarities)/np.sqrt(len(similarities))

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

def angle_change(pre_space, post_space):
    """
    for two given trajectories, computes the change in the angle between the hyperplanes that best explain them
    """
    fit, residual = plane_of_best_fit(pre_space)
    pre_norm = (-fit[0], -fit[1], 1) # normal vector to the plane of best fit
    fit, residual = plane_of_best_fit(post_space)
    post_norm = (-fit[0], -fit[1], 1) # normal vector to the plane of best fit
    angle = np.arccos(np.dot(pre_norm, post_norm)/(np.linalg.norm(pre_norm)*np.linalg.norm(post_norm)))
    angle = np.degrees(np.real(angle))[0]
    return angle

def paired_procrustes_similarity(pre_spaces, post_spaces):
    """
    Computes the averaged proscrutes similarity between the pre and post lesion spaces across all iterations
    """
    similarities = []
    angle_changes = []
    for i in range(len(pre_spaces)):
        pre_space = pre_spaces[i]
        post_space = post_spaces[i]
        procrustes_similarity = spatial.procrustes(pre_space, post_space)[2]
        similarities.append(procrustes_similarity)

        angle = angle_change(pre_space, post_space)
        angle_changes.append(angle)

    return np.average(similarities), np.std(similarities), np.std(similarities)/np.sqrt(len(similarities)), np.average(angle_changes)

pre_average, pre_std, pre_sem = pairwise_procrustes_similarity(pre_spaces)
post_average, post_std, post_sem, avg_del_angle = paired_procrustes_similarity(pre_spaces, post_spaces)

mse_path = os.path.join(path, 'pre_procrustes')  # Main directory
if not os.path.exists(mse_path):
    os.makedirs(mse_path)

file_path = os.path.join(mse_path,f"job_{ARRAY_ID}_E:I_{rel_e_i:2f}.csv")

with open(file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['ratio','average', 'std', 'sem'])
    writer.writerow([rel_e_i, pre_average, pre_std, pre_sem])

mse_path = os.path.join(path, 'post_procrustes')  # Main directory
if not os.path.exists(mse_path):
    os.makedirs(mse_path)

file_path = os.path.join(mse_path,f"job_{ARRAY_ID}_E:I_{rel_e_i:2f}.csv")

with open(file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['ratio','average', 'std', 'sem', 'avg_del_angle'])
    writer.writerow([rel_e_i, post_average, post_std, post_sem, avg_del_angle])

# mse_path = os.path.join(path, 'angle_change')  # Main directory
# if not os.path.exists(mse_path):
#     os.makedirs(mse_path)

# file_path = os.path.join(mse_path,f"job_{ARRAY_ID}_E:I_{rel_e_i:2f}.csv")
# with open(file_path, 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(['ratio','average', 'std', 'sem'])
#     writer.writerow([rel_e_i, post_average, post_std, post_sem])
