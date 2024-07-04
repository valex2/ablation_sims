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

e_i_ratio_range = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] # TODO this is the parameter space that we range across
ablation_frac_range = np.linspace(0, 1, 25)
rel_e_i = e_i_ratio_range[ARRAY_ID] # pull this based off of the run ID
NUM_TOTAL = 100 # number of total neurons for rel e:i
num_exc = int(np.ceil(NUM_TOTAL * rel_e_i))
num_inh = int(np.ceil(NUM_TOTAL * (1 - rel_e_i)))

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

null_mses = {}
conn_mses = {}
for ablation_frac in ablation_frac_range:
    print(f"{ablation_frac} out of {ablation_frac_range}")
    null_iter = []
    conn_iter = []
    for iter in range(1):
        print(f"iter num {iter}")
        with model: 
            ### population initialization ###
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

        with nengo.Simulator(model) as sim:
            ##### NOTE: pre-ablation run #####
            sim.run(sim_duration)
            t = sim.trange()
            pre_ablation_e_probe = sim.data[e_probe][10:, :] # value at each milisecond

            ##### NOTE: null population metrics #####
            null_vals = sim.data[null_probe][10:,:] # null values

            #------------------------------------------------------------------------------------------------------------------------#
        
            losers = ablate_population([exc_pop, inh_pop], ablation_frac, sim)
            ablate_population([null_pop], ablation_frac, sim)

            ##### NOTE: post-ablation run #####
            sim.run(sim_duration)
            t = sim.trange()
            post_ablation_e_probe = sim.data[e_probe][sim_duration*1000 + 10:, :] # value at each milisecond
            post_ablation_null_probe = sim.data[null_probe][sim_duration*1000 + 10:, :]
            
            #------------------------------------------------------------------------------------------------------------------------#

            ##### NOTE: output calculations #####
            null_iter.append(calculate_mse(null_vals, post_ablation_null_probe))
            conn_iter.append(calculate_mse(pre_ablation_e_probe, post_ablation_e_probe))
            
            # print(f"nulls: {calculate_mse(layered_periodic_function(t[10:(sim_duration * 1000)]), null_vals)}")
            # print(f"pre: {calculate_mse(layered_periodic_function(t[10:(sim_duration * 1000)]), pre_ablation_e_probe)}")
            # print(f"post: {calculate_mse(layered_periodic_function(t[10:(sim_duration * 1000)]), post_ablation_e_probe)}")
        null_mses[ablation_frac] = null_iter
        conn_mses[ablation_frac] = conn_iter

mse_path = os.path.join(path, 'null_mse_4')  # Main directory
if not os.path.exists(mse_path):
    os.makedirs(mse_path)

file_path = os.path.join(mse_path,f"job_{ARRAY_ID}_E:I_{rel_e_i:2f}.csv")

with open(file_path, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=null_mses.keys())
    
    writer.writeheader() # Write the header
    
    for row in zip(*null_mses.values()): # Write the rows
        writer.writerow(dict(zip(null_mses.keys(), row)))

mse_path = os.path.join(path, 'conn_mse_4')  # Main directory
if not os.path.exists(mse_path):
    os.makedirs(mse_path)

file_path = os.path.join(mse_path,f"job_{ARRAY_ID}_E:I_{rel_e_i:2f}.csv")

with open(file_path, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=conn_mses.keys())
    
    writer.writeheader() # Write the header
    
    for row in zip(*conn_mses.values()): # Write the rows
        writer.writerow(dict(zip(conn_mses.keys(), row)))
