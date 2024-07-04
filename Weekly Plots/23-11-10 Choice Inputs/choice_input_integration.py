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

model = nengo.Network(label="E/I offset")

##### NOTE: CREATE SIMULATION FOLDER AND CONFIG #####
os.environ
OUT_DIR = "/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Izkevich Analysis" #NOTE: change this for output folder
os.system('mkdir {}'.format(os.path.join(OUT_DIR,'plots')))

name = "Checking Gain Biasing"

path = os.path.join(OUT_DIR,f'{name} plots')
if not os.path.exists(path):
        os.makedirs(path)

### PARMATERS YAML ###
runID_num = '01'
slurm_dict =  {
              'num_exc_neur' : 90, # 360
              'num_inh_neur' : 10, # 40
              'ablation_frac' : 0.50, # fraction of neurons to lesion
              
              'e_i_precent_conn' : 0.05,#0.05, # excitatory connectivity ratio
              'e_i_max_val' : 0.05, # the greatest strength of the e_i connection
              'e_i_tailed_dist': True, # whether excitatory connection strengths are drawn from a tailed distribution (with max val) or not

              'i_e_precent_conn' : 0.05,#0.05, # inhibitory connectivity ratio
              'i_e_max_val' : -0.1, # the greatest strength of the i_e connection
              'i_e_tailed_dist': True, # whether inhibitory connection strengths are drawn from a tailed distribution (with max val) or not
              
              'i_i_precent_conn' : 0.05, # recurrent inhibitory connectivity ratio
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
ablation_frac = slurm_dict['ablation_frac']
time_cut_off = slurm_dict['time_cut_off']
lesion_bars = slurm_dict['lesion_bars']

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

# NOTE: given a precent connectivity, outputs corresponding transform matrix
# inputs: 
# connection_weight -- the degree of neuron-to-neuron weighting, make negative for inhibitory connections
# precent -- measure of connectivity between populations
# size... -- sizes of pre/post connection populations
# output: m x n transform matrix mapping connectivity with desired precentage
def sparse_connectivity(precent, connection_weight, size_conn_from, size_conn_to, tailed_dist): 
    transform_entries = size_conn_to * size_conn_from
    desired_num_connections = math.ceil(transform_entries * precent) # utilizing ceiling to avoid 0-connected matrices
    inhibitory_transform = np.zeros((size_conn_to, size_conn_from)) # 0 matrix of (num_e x num_i)
    dist = np.linspace(connection_weight, 0, 100) # for distribution tailing
    connected = 0 # tracking the num of connections formed in transform matrix 
    while connected < desired_num_connections:
        row = random.randint(0, size_conn_to - 1)
        col = random.randint(0, size_conn_from - 1)
        if inhibitory_transform[row][col] == 0: 
            if tailed_dist:
                inhibitory_transform[row][col] = np.random.choice(dist)
            else:
                inhibitory_transform[row][col] = connection_weight # negative 1 for inhibitory transformation
            connected = connected + 1
    return inhibitory_transform # output the transform matrix

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
        spread_metric = (data[:10] @ n) / 100
        spread_metrics.append(spread_metric)
        
        x_pos = np.arange(num_eigs) + i * bar_width  # Adjust x_pos for each group of bars
        bars = ax.bar(x_pos, data[:num_eigs], color=color, edgecolor='black', linewidth=1, width=bar_width, zorder=2, label=f"{modifier}, σ={spread_metric:.2f}")
        bars_list.append(bars)
    
    total_bars = len(data_list)
    x_pos_middle = np.arange(num_eigs) + (total_bars - 1) * bar_width / 2
    ax.set_xticks(x_pos_middle)
    ax.set_xticklabels([f'PC{i}' for i in range(num_eigs)], fontsize=10, rotation=90, ha='center')
    ax.set_ylabel('percentage of variance explained', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    for bars, modifier, color in zip(bars_list, modifier_list, colors):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 4),
                        textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold', color=color, rotation = "vertical")
    
    fig.tight_layout()
    return fig, spread_metrics

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

def plot_network(conn_matrix, blackout_nodes=[]):
    G = nx.from_numpy_array(conn_matrix)
    graph_plot = plt.figure(figsize=(10, 8))
    
    # Create separate lists for red and green nodes
    red_nodes = [node for node in G.nodes() if node > num_exc_neur]
    green_nodes = [node for node in G.nodes() if node <= num_exc_neur]
    
    # Calculate positions for red nodes on the left and green nodes on the right
    pos = {}
    for node in red_nodes:
        pos[node] = (np.random.normal(1, 0.1), np.random.uniform(-2, 2))  # Position on the left
    for node in green_nodes:
        pos[node] = (np.random.normal(-1, 0.1), np.random.uniform(-3, 3))  # Position on the right

    # Extract edge weights from the matrix
    edge_weights = [conn_matrix[edge[0]][edge[1]] for edge in G.edges()]

    # Normalize edge weights for color mapping
    min_weight = np.min(edge_weights)
    max_weight = np.max(edge_weights)

    edge_colors = []

    for edge in G.edges():
        source, target = edge
        weight = conn_matrix[source][target]

        # Check if either the source or target node is a blackout node
        if source in blackout_nodes or target in blackout_nodes:
            edge_color = (0, 0, 0, 0.3)  # Black for connections involving blackout nodes
        else:
            norm_weight = (weight - min_weight) / (max_weight - min_weight)
            if weight < 0:
                edge_color = mcolors.to_rgba((1, 0, 0, norm_weight * 0.6))  # Red for negative values (pastel)
            else:
                edge_color = mcolors.to_rgba((0, 1, 0, norm_weight * 0.6))  # Green for positive values (pastel)
        edge_colors.append(edge_color)
        
    # Define node_colors based on red and green nodes, with blackout support
    node_colors = ["red" if node in red_nodes else "green" for node in G.nodes()]
    for node in blackout_nodes:
        node_colors[node] = "black"
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Excitatory Neurons'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Inhibitory Neurons'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Lesioned Neurons')
    ]
    plt.legend(handles=legend_elements, loc='upper center')
    plt.title("Example Sparse E:I Connectivity Map", fontsize=14, fontweight='bold')
    nx.draw(G, pos, with_labels=False, node_size=50, width=2,
            node_color=node_colors, edge_color=edge_colors)
    plt.savefig(path + "/network", dpi = 300)
    return graph_plot

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

#### NOTE: this function computes the mean square error between two sets of data (the data must have the same length)
def calculate_mse(baseline, for_comparison):
    if len(for_comparison) != len(baseline):
        raise ValueError("Signals must have the same length")
    
    squared_diff = np.square(baseline - for_comparison)
    mse = np.mean(squared_diff)
    return mse

# def calculate_mse(baseline, for_comparison):
#     euclidean_distance = np.sqrt(np.sum((baseline - for_comparison)**2)) / len(baseline)
#     return euclidean_distance

with model: 
    ### population initialization ###
    rand_num = np.random.randint(0, 4000) # seed the null and exc population such that they're the same
    exc_pop = nengo.Ensemble(n_neurons = num_exc_neur, dimensions = 1, seed = rand_num, neuron_type=nengo.Izhikevich(tau_recovery=0.15)) ## neuron_type=nengo.Izhikevich(tau_recovery=0.1)
    inh_pop = nengo.Ensemble(n_neurons = num_inh_neur, dimensions = 1)
    null_pop = nengo.Ensemble(n_neurons = num_exc_neur, dimensions = 1, seed = rand_num, neuron_type=nengo.Izhikevich(tau_recovery=0.15)) # neuron_type=nengo.Izhikevich(tau_recovery=0.1) # baseline to see whether signal is reproducible given population size
    #intermediate = nengo.Ensemble(n_neurons = 20, dimensions = 5)

    input_signal = nengo.Node(output=layered_periodic_function)
    # input_signal = nengo.Node(nengo.processes.WhiteSignal(1.0, high=50, seed=0))
    # input_signal = nengo.Node(nengo.processes.WhiteNoise(dist = nengo.dists.Gaussian(0,1)))

    ### connectivity ###
    # NOTE: a transform is a linear transformation  mapping the pre function output to the post function input
    conn, e_e, e_i, i_e, i_i = balance_condition(num_exc_neur, num_inh_neur, e_i_precent_conn, i_e_precent_conn)

    # inhibitory_transform_i_e = sparse_connectivity(i_e_precent_conn, i_e_max_val, inh_pop.n_neurons, exc_pop.n_neurons, i_e_tailed_dist) # -1 since inhibitory
    # inhibitory_transform_i_i = sparse_connectivity(i_i_precent_conn, i_i_max_val, inh_pop.n_neurons, inh_pop.n_neurons, i_i_tailed_dist) # -1 since inhibitory
    # # excitatory_transform = [[1] * num_exc_neur for n in range(num_inh_neur)] # the excitatory transform is fully connected
    # excitatory_transform = sparse_connectivity(e_i_precent_conn, e_i_max_val, exc_pop.n_neurons, inh_pop.n_neurons, e_i_tailed_dist)
    nengo.Connection(input_signal, exc_pop) #input to excitatory
    nengo.Connection(exc_pop.neurons, exc_pop.neurons, transform = e_e)
    nengo.Connection(exc_pop.neurons, inh_pop.neurons, transform = e_i) # network connections
    nengo.Connection(inh_pop.neurons, exc_pop.neurons, transform = i_e)
    nengo.Connection(inh_pop.neurons, inh_pop.neurons, transform = i_i)
    nengo.Connection(input_signal, null_pop) # connect to the null_pop
    
    ### probing ###
    e_probe = nengo.Probe(exc_pop, synapse=probe_synapse)
    i_probe = nengo.Probe(inh_pop, synapse=probe_synapse)
    e_voltage = nengo.Probe(exc_pop.neurons, "voltage", synapse=probe_synapse)
    i_voltage = nengo.Probe(inh_pop.neurons, "voltage", synapse=probe_synapse)
    e_spikes = nengo.Probe(exc_pop.neurons)
    i_spikes = nengo.Probe(inh_pop.neurons)
    null_probe = nengo.Probe(null_pop, synapse=probe_synapse)
    null_spikes = nengo.Probe(null_pop.neurons)

with nengo.Simulator(model) as sim:
    ##### NOTE: pre-ablation run #####
    sim.run(sim_duration)
    t = sim.trange()
    e_spk = sim.data[e_spikes][10:, :] # spiking at each milisecond -- (length of sim (ms) x num_neurons)
    pre_ablation_e_probe = sim.data[e_probe][10:, :] # value at each milisecond
    pre_binary = e_spk.T # (num_neurons x length of sim (ms))
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

    ##### NOTE: joint population metrics, pre-ablation #####
    i_spk = sim.data[i_spikes][10:, :]
    pre_i_binary = i_spk.T
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

    pre_e = pre_binary # (num_neurons x length of sim (ms))
    pre_i = pre_i_binary # (num_neurons x length of sim (ms))

    ##### NOTE: null population metrics #####
    null_vals = sim.data[null_probe][10:,:] # null values
    null_spk = sim.data[null_spikes][10:,:] # null_spikes
    null_binary = null_spk.T
    null_n_bin = int(len(null_binary[0,:])/t_bin) # num bins required given bin time size
    null_spike_count = np.zeros((len(null_binary),null_n_bin)) # initialize spike count tracker -- (num_neurons x (ms / t_bin)) == (num_neurons x (n_bin))
    for j in range(n_bin - 1):
        null_spike_count[:,j] = null_binary[:,j*t_bin:(j+1)*t_bin].sum(axis=1) # populate spike counts given bin size
    Y = null_spike_count.transpose()
    Y = StandardScaler().fit_transform(Y)
    null_cov_mat = np.cov(Y, rowvar=False) # (nueron x neuron) covariance matrix based on inner product across all dimensions 
    null_eig_vals, null_eig_vect = np.linalg.eig(null_cov_mat) # eigenvalues = (num_neurons x 1), eigenvectors = (num_neurons x num_neurons) -- each column is an eigenvector
    null_eig_sum = np.sum(np.abs(null_eig_vals)) # sum up all the eigenvalues
    null_povs = np.sort(100*np.abs(null_eig_vals)/null_eig_sum)[::-1] # for later plotting

    ##### NOTE: SVD power plots, null #####
    U, S, VT = np.linalg.svd(Y, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    S_null = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else

    ##### NOTE: SVD power plots, pre-ablation #####
    U, S, VT = np.linalg.svd(X, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    S_pre_ablation = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else

    ##### NOTE: SVD power plots, pre-ablation, joint #####
    U, S, VT = np.linalg.svd(T, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    S_joint_pre_ablation = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else

    #------------------------------------------------------------------------------------------------------------------------#
   
    ##### NOTE: performing ablation #####
    ensembles = []
    if lesion_e:
        ensembles.append(exc_pop)
    if lesion_i:
        ensembles.append(inh_pop)

    losers = ablate_population(ensembles, ablation_frac, sim)
    ablate_population([null_pop], ablation_frac, sim)

    ##### NOTE: post-ablation run #####
    sim.run(sim_duration)
    t = sim.trange()
    print(f"\nsim duration: {sim_duration}, t (sim_range) = {t}\n")
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
    post_eig_sum = np.sum(np.abs(post_eig_vals)) # sum up all the eigenvalues
    povs_post = np.sort(100*np.abs(post_eig_vals)/post_eig_sum)[::-1] # save for later

    ##### NOTE: joint population metrics, post-ablation #####
    i_spk = sim.data[i_spikes][sim_duration*1000 + 10:, :]
    post_i_binary = i_spk.T
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

    #------------------------------------------------------------------------------------------------------------------------#

    ##### NOTE: state space plot initialization, exc pop #####
    exc_state_space = plt.figure() # Create a 3D plot
    ax = exc_state_space.add_subplot(111, projection='3d')
    ax.set_xlabel('PC 1', labelpad=-2) # Set labels for the axes
    ax.set_ylabel('PC 2', labelpad=-2)
    ax.set_zlabel('PC 3', labelpad=-2)
    ax.set_xticks([]) # remove the tick marks from the axis
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("State Space", fontsize=14, fontweight='bold')
    cmap = plt.get_cmap('Blues')

    ##### NOTE: state space plotting, pre-ablation #####
    idx = pre_eig_vals.argsort()[::-1] # sort the eigenvalues and eigenvectors from highest to lowest  
    pre_eig_vals = pre_eig_vals[idx] # eigval shape is (40,)
    pre_eig_vect = pre_eig_vect[:,idx] # eigvector shape is (40 x 40) where each column represents an eigenvector
    eig3D = pre_eig_vect[:, :3]
    neural_state_space = pre_e_spike_count.T @ eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space
    num_time_steps = neural_state_space.shape[0] # Create a colormap based on the number of time steps
    for i in range(num_time_steps): # plot the points with gradient coloring
        x = neural_state_space[i, 0]
        y = neural_state_space[i, 1]
        z = neural_state_space[i, 2]
        ax.scatter(x, y, z, c=[cmap(i / num_time_steps)], s=10)  # Adjust s for point size

    ##### NOTE: plane plotting #####
    fit, residual = plane_of_best_fit(neural_state_space)
    xmin, xmax = np.min(neural_state_space[:, 0]), np.max(neural_state_space[:, 0]) # specify linspace overwhich to plot the plane based on datapoints
    ymin, ymax = np.min(neural_state_space[:, 1]), np.max(neural_state_space[:, 1])
    xx, yy = np.meshgrid(np.linspace(xmin, xmax), np.linspace(ymin, ymax)) # define the grid
    pre_z = (fit[0] * xx + fit[1] * yy + fit[2])
    pre_norm = (-fit[0], -fit[1], 1)
    ax.plot_surface(xx, yy, pre_z, alpha = 0.3, color = "lightblue") # plot the plane
    marker = line_integral(neural_state_space)
    ax.plot(neural_state_space[0:(num_time_steps - 1),0], neural_state_space[0:(num_time_steps - 1),1], neural_state_space[0:(num_time_steps - 1),2], c = "blue", label = f"pre-ablation trajectory, λ={marker:.2f}", alpha = 0.4) # plot the trajectory

    ##### NOTE: state space plotting, post-ablation #####
    ### -- we're projecting the ablated data into the same eigenspace as the healthy data for comparison purposes -- ###
    neural_state_space = post_e_spike_count.T @ eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space
    cmap = plt.get_cmap('inferno')
    for i in range(num_time_steps): # plot the points with gradient coloring
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
    ax.plot(neural_state_space[0:(num_time_steps - 1),0], neural_state_space[0:(num_time_steps - 1),1], neural_state_space[0:(num_time_steps - 1),2], c = "red", label = f"post-ablation trajectory, λ={marker:.2f}, θ={del_angle_degrees:.2f}º", alpha = 0.4) # plot the trajectory

    ##### NOTE: making the state space plot look nice #####
    plt.legend()
    ax.view_init(elev=33, azim=11) # rotate the plot
    exc_state_space.savefig(path + "/exc_state_space_rot_1", dpi = 300)
    ax.view_init(elev=46, azim=138) # rotate the plot
    exc_state_space.savefig(path + "/exc_state_space_rot_2", dpi = 300)

    #------------------------------------------------------------------------------------------------------------------------#

    ##### NOTE: state space plot initialization, whole pop #####
    whole_state_space = plt.figure() # Create a 3D plot
    ax = whole_state_space.add_subplot(111, projection='3d')
    ax.set_xlabel('PC 1', labelpad=-2) # Set labels for the axes
    ax.set_ylabel('PC 3', labelpad=-2)
    ax.set_zlabel('PC 2', labelpad=-2)
    ax.set_xticks([]) # remove the tick marks from the axis
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("pre/post ablation state space, whole pop")
    cmap = plt.get_cmap('Blues')

    ##### NOTE: state space plotting, pre-ablation, whole pop #####
    idx = pre_joint_eig_vals.argsort()[::-1] # sort the eigenvalues and eigenvectors from highest to lowest  
    pre_joint_eig_vals = pre_joint_eig_vals[idx] # eigval shape is (40,)
    pre_joint_eig_vect = pre_joint_eig_vect[:,idx] # eigvector shape is (40 x 40) where each column represents an eigenvector
    joint_eig3D = pre_joint_eig_vect[:, :3]
    neural_state_space = pre_joint_spike_count.T @ joint_eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space
    neural_state_space = np.real(neural_state_space) ## TODO
    reorder = [0, 2, 1]
    neural_state_space = neural_state_space[:, reorder]
    num_time_steps = neural_state_space.shape[0] # Create a colormap based on the number of time steps
    for i in range(num_time_steps): # plot the points with gradient coloring
        x = neural_state_space[i, 0]
        y = neural_state_space[i, 1]
        z = neural_state_space[i, 2]
        ax.scatter(x, y, z, c=[cmap(i / num_time_steps)], s=10)  # Adjust s for point size
    marker = line_integral(neural_state_space)
    ax.plot(neural_state_space[0:(num_time_steps - 1),0], neural_state_space[0:(num_time_steps - 1),1], neural_state_space[0:(num_time_steps - 1),2], c = "blue", label = f"pre-ablation trajectory, λ={marker:.3f}", alpha = 0.4) # plot the trajectory

    ##### NOTE: plane plotting #####
    fit, residual = plane_of_best_fit(neural_state_space)
    xmin, xmax = np.min(neural_state_space[:, 0]), np.max(neural_state_space[:, 0]) # specify linspace overwhich to plot the plane based on datapoints
    ymin, ymax = np.min(neural_state_space[:, 1]), np.max(neural_state_space[:, 1])
    xx, yy = np.meshgrid(np.linspace(xmin, xmax), np.linspace(ymin, ymax)) # define the grid
    pre_z = (fit[0] * xx + fit[1] * yy + fit[2])
    pre_norm = (-fit[0], -fit[1], 1)
    ax.plot_surface(xx, yy, pre_z, alpha = 0.3, color = "lightblue") # plot the plane

    ##### NOTE: state space plotting, post-ablation, whole pop #####
    neural_state_space = post_joint_spike_count.T @ joint_eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space
    neural_state_space = np.real(neural_state_space) ## TODO
    neural_state_space = neural_state_space[:, reorder]
    num_time_steps = neural_state_space.shape[0] # Create a colormap based on the number of time steps
    cmap = plt.get_cmap('inferno')
    for i in range(num_time_steps): # plot the points with gradient coloring
        x = neural_state_space[i, 0]
        y = neural_state_space[i, 1]
        z = neural_state_space[i, 2]
        ax.scatter(x, y, z, c=[cmap(i / num_time_steps)], s=10)  # Adjust s for point size

    ##### NOTE: plane plotting #####
    fit, residual = plane_of_best_fit(neural_state_space)
    post_z = (fit[0] * xx + fit[1] * yy + fit[2])
    post_norm = (-fit[0], -fit[1], 1)
    ax.plot_surface(xx, yy, post_z, alpha = 0.3, color = "lightcoral") # plot the plane

    ##### NOTE: plane metrics #####
    joint_del_angle_radians = np.arccos(np.dot(pre_norm, post_norm) / (np.linalg.norm(pre_norm) * np.linalg.norm(post_norm)))
    joint_del_angle_degrees = np.degrees(joint_del_angle_radians)[0]
    marker = line_integral(neural_state_space)
    ax.plot(neural_state_space[0:(num_time_steps - 1),0], neural_state_space[0:(num_time_steps - 1),1], neural_state_space[0:(num_time_steps - 1),2], c = "red", label = f"post-ablation trajectory, λ={marker:.2f}, θ={joint_del_angle_degrees:.2f}", alpha = 0.4) # plot the trajectory

    ##### NOTE: making the state space plot look nice #####
    plt.legend()
    ax.view_init(elev=33, azim=11) # rotate the plot
    whole_state_space.savefig(path + "/whole_state_space_rot_1", dpi = 300)
    ax.view_init(elev=46, azim=138) # rotate the plot
    whole_state_space.savefig(path + "/whole_state_space_rot_2", dpi = 300)

    #------------------------------------------------------------------------------------------------------------------------#

    ##### NOTE: state space plot initialization, null pop #####
    null_state_space = plt.figure() # Create a 3D plot
    ax = null_state_space.add_subplot(111, projection='3d')
    ax.set_xlabel('PC 1', labelpad=-2) # Set labels for the axes
    ax.set_ylabel('PC 2', labelpad=-2)
    ax.set_zlabel('PC 3', labelpad=-2)
    ax.set_xticks([]) # remove the tick marks from the axis
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("null pop state space")
    cmap = plt.get_cmap('Greens')

    ##### NOTE: state space plotting null pop #####
    idx = null_eig_vals.argsort()[::-1] # sort the eigenvalues and eigenvectors from highest to lowest  
    null_eig_vals = null_eig_vals[idx] # eigval shape is (40,)
    null_eig_vect = null_eig_vect[:,idx] # eigvector shape is (40 x 40) where each column represents an eigenvector
    null_eig3D = null_eig_vect[:, :3]
    neural_state_space = null_spike_count.T @ null_eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space
    num_time_steps = neural_state_space.shape[0] # Create a colormap based on the number of time steps
    for i in range(num_time_steps): # plot the points with gradient coloring
        x = neural_state_space[i, 0]
        y = neural_state_space[i, 1]
        z = neural_state_space[i, 2]
        ax.scatter(x, y, z, c=[cmap(i / num_time_steps)], s=10)  # Adjust s for point size
    ax.plot(neural_state_space[0:(num_time_steps - 1),0], neural_state_space[0:(num_time_steps - 1),1], neural_state_space[0:(num_time_steps - 1),2], c = "green", label = "null trajectory", alpha = 0.4) # plo

    ##### NOTE: making the state space plot look nice #####
    plt.legend()
    ax.view_init(elev=33, azim=11) # rotate the plot
    null_state_space.savefig(path + "/null_state_space_rot_1", dpi = 300)
    ax.view_init(elev=46, azim=138) # rotate the plot
    null_state_space.savefig(path + "/null_state_space_rot_2", dpi = 300)

    #------------------------------------------------------------------------------------------------------------------------#

    ##### NOTE: SVD power plots, post-ablation #####
    U, S, VT = np.linalg.svd(X, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    S_post_ablation = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else

    ##### NOTE: SVD power plots, post-ablation, joint #####
    U, S, VT = np.linalg.svd(T, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    S_joint_post_ablation = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else
    
    ##### NOTE: POV plotting, excitatory population only #####
    together_POV, together_var = plot_POVs([null_povs, povs_pre, povs_post], [f"null", f"pre-ablation, {i_e_precent_conn*100}% connected", f"post-ablation, {i_e_precent_conn*100}% connected"], ["lightgreen", "lightblue", "lightcoral"], f"excitatory pop. variance explained by top {num_eigsum} PCs")
    together_POV.savefig(path + "/together_POV", dpi = 300)

    ##### NOTE: POV plotting, entire population #####
    pop_POV, pop_var = plot_POVs([joint_povs_pre, joint_povs_post], [f"pre-ablation, {i_e_precent_conn*100}% connected", f"post-ablation, {i_e_precent_conn*100}% connected"], ["lightblue", "lightcoral"], f"whole pop. variance explained by top {num_eigsum} PCs")
    pop_POV.savefig(path + "/pop_POV", dpi = 300)
    
    ##### NOTE: inter-connection connectivity plotting #####

    ##### NOTE: SV/power plotting
    sv_plot = plt.figure() # singular value plotting
    plt.semilogy(np.diag(S_null), linestyle = "dashed", label="null", color = "lightgreen")
    plt.semilogy(np.diag(S_pre_ablation), linestyle = "dashed", label="exc pop, pre-ablation", color = "lightblue")
    plt.semilogy(np.diag(S_post_ablation), linestyle = "dashed", label="exc pop, post-ablation", color = "lightcoral")
    plt.semilogy(np.diag(S_joint_pre_ablation), linestyle = "solid", label="whole pop, pre-ablation", color = "lightblue")
    plt.semilogy(np.diag(S_joint_post_ablation), linestyle = "solid", label="whole pop, post-ablation", color = "lightcoral")
    plt.title('SV Log-Scale Power')
    plt.xlabel("singular value #")
    plt.ylabel("power of singular value")
    plt.legend()
    sv_plot.savefig(path + "/sv_plot", dpi = 300)

    sv_cum_contribution = plt.figure()
    plt.plot(np.cumsum((np.diag(S_null))/np.sum(np.diag(S_null)))*100, linestyle = "dashed", label = "null", color = "lightgreen")
    plt.plot(np.cumsum((np.diag(S_pre_ablation))/np.sum(np.diag(S_pre_ablation)))*100, linestyle = "dashed", label = "exc pop, pre-ablation", color = "lightblue")
    plt.plot(np.cumsum((np.diag(S_post_ablation))/np.sum(np.diag(S_post_ablation)))*100, linestyle = "dashed", label = "exc pop, post-ablation", color = "lightcoral")
    plt.plot(np.cumsum((np.diag(S_joint_pre_ablation))/np.sum(np.diag(S_joint_pre_ablation)))*100, linestyle = "solid", label = "whole pop, pre-ablation", color = "lightblue")
    plt.plot(np.cumsum((np.diag(S_joint_post_ablation))/np.sum(np.diag(S_joint_post_ablation)))*100, linestyle = "solid", label = "whole pop, post-ablation", color = "lightcoral")
    plt.title('SV Cumulative Contribution')
    plt.xlabel("# of singular values")
    plt.ylabel("percent contribution to overall data")
    plt.legend()
    sv_cum_contribution.savefig(path + "/sv_cum_contribution", dpi = 300)

    ##### NOTE: e-i connectivity plotting #####
    plt.figure()
    max = np.abs(e_i_max_val)
    sns.heatmap(e_i, annot=False, vmin = -max, vmax = max, cmap = cm.RdBu)
    plt.ylabel("inhibitory population neurons")
    plt.xlabel("excitatory population neurons")
    plt.title("e->i Connectivity Map")
    plt.savefig(path + "/e_i_graph_output", dpi = 300)

    ##### NOTE: intra-connection connectivity plotting #####
    plt.figure()
    max = np.abs(i_e_max_val)
    sns.heatmap(i_e, annot=False, vmin = -max, vmax = max, cmap = cm.RdBu)
    plt.ylabel("excitatory population neurons")
    plt.xlabel("inhibitory population neurons")
    plt.title("i->e Connectivity Map")
    plt.savefig(path + "/i_e_graph_output", dpi = 300)

    ##### NOTE: recurrent-connection connectivity plotting #####
    plt.figure()
    max = np.abs(i_i_max_val)
    sns.heatmap(i_i, annot=False, vmin = -max, vmax = max, cmap = cm.RdBu)
    plt.ylabel("inhibitory population neurons")
    plt.xlabel("inhibitory population neurons")
    plt.title("i->i Connectivity Map")
    plt.savefig(path + "/i_i_graph_output", dpi = 300)

    ##### NOTE: full-connectivity plotting #####
    plt.figure()
    vmax = np.max(abs(conn))
    sns.heatmap(conn, annot=False, vmin = -vmax, vmax = vmax, cmap = 'seismic_r')
    # plt.show()

    plot_network(conn, losers)

    ##### NOTE: output plotting #####
    plt.figure()
    plt.plot(t[10:(sim_duration * 1000)], layered_periodic_function(t[10:(sim_duration * 1000)]), label = f"input", color = "slateblue", alpha = 0.4)
    plt.plot(t[10:(sim_duration * 1000)], null_vals, label = f"excitatory output", color = "darkgreen")
    plt.plot(t[10:(sim_duration * 1000)], post_ablation_null_probe, label = f"excitatory,{ablation_frac*100}% ablated output", color = "b")
    plt.plot(t[10:(sim_duration * 1000)], pre_ablation_e_probe, label = f"{i_e_precent_conn*100}% connected output", color = "darkorange")
    plt.plot(t[10:(sim_duration * 1000)], post_ablation_e_probe, label = f"{i_e_precent_conn*100}% connected, {ablation_frac*100}% ablated output", color = "r")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("I/O Relationship", fontsize=14, fontweight='bold')
    plt.legend()
    plt.savefig(path + "/IO_relationship", dpi = 300)
    print(f"nulls: {calculate_mse(layered_periodic_function(t[10:(sim_duration * 1000)]), null_vals)}")
    print(f"pre: {calculate_mse(layered_periodic_function(t[10:(sim_duration * 1000)]), pre_ablation_e_probe)}")
    print(f"post: {calculate_mse(layered_periodic_function(t[10:(sim_duration * 1000)]), post_ablation_e_probe)}")
    print(f"null: {calculate_mse(null_vals, post_ablation_null_probe)}")
    print(f"connected: {calculate_mse(pre_ablation_e_probe, post_ablation_e_probe)}")

    ##### NOTE: pre-ablation raster plotting #####
    fig, ax = plt.subplots()
    pre_joint_binary = pre_joint_binary[:, :time_cut_off]
    pre_spike_rates = np.sum(pre_joint_binary, axis=1, keepdims=True) # sum across the time domain
    pre_exc_rate = np.sum(pre_spike_rates[:num_exc_neur]) / (time_cut_off * num_exc_neur) # average across time
    pre_inh_rate = np.sum(pre_spike_rates[num_exc_neur : num_exc_neur+num_inh_neur]) / (time_cut_off * num_inh_neur) # average across time
    for neuron_idx in range(num_exc_neur + num_inh_neur):
        spike_times = np.where(pre_joint_binary[neuron_idx])[0]
        ax.scatter(spike_times, np.full_like(spike_times, neuron_idx), marker='|', color='green' if neuron_idx < num_exc_neur else 'red', s=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron')
    ax.set_title('Pre-Ablation', fontsize=14, fontweight='bold')
    ax.axhspan(0, num_exc_neur, facecolor='green', alpha=0.05, label='excitatory neurons')
    ax.axhspan(num_exc_neur, num_exc_neur + num_inh_neur, facecolor='red', alpha=0.05, label='inhibitory neurons')
    ax.set_ylim(-1, num_exc_neur + num_inh_neur)
    # exc_legend = ax.scatter([], [], marker='|', color='g', label=f'exc | spikes/ms: {pre_exc_rate:.2f}')
    # inh_legend = ax.scatter([], [], marker='|', color='red', label=f'inh | spikes/ms: {pre_inh_rate:.2f}')
    # ax.legend(handles=[inh_legend, exc_legend], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.savefig(path + "/pre_ablation_spike_raster", dpi = 300)

    ##### NOTE: post-ablation raster plotting #####
    fig, ax = plt.subplots()
    post_joint_binary = post_joint_binary[:, :time_cut_off]
    post_spike_rates = np.sum(post_joint_binary, axis=1, keepdims=True) # sum across the time domain
    post_exc_rate = np.sum(post_spike_rates[:num_exc_neur]) / (time_cut_off * num_exc_neur) # average across time
    post_inh_rate = np.sum(post_spike_rates[num_exc_neur : num_exc_neur+num_inh_neur]) / (time_cut_off * num_inh_neur) # average across time
    for neuron_idx in range(num_exc_neur + num_inh_neur):
        spike_times = np.where(post_joint_binary[neuron_idx])[0]
        ax.scatter(spike_times, np.full_like(spike_times, neuron_idx), marker='|', color='g' if neuron_idx < num_exc_neur else 'red', s=0.5)
    if lesion_bars:
        for loser_y in losers: # show which neurons were ablated
            ax.axhline(y=loser_y, color='black', linewidth=1, alpha = 0.5) 
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron')
    ax.set_title('Post-Ablation', fontsize=14, fontweight='bold')
    ax.axhspan(0, num_exc_neur, facecolor='green', alpha=0.05, label='excitatory neurons')
    ax.axhspan(num_exc_neur, num_exc_neur + num_inh_neur, facecolor='red', alpha=0.05, label='inhibitory neurons')
    ax.set_ylim(-1, num_exc_neur + num_inh_neur)
    # exc_legend = ax.scatter([], [], marker='|', color='g', label=f'exc | spikes/ms: {post_exc_rate:.2f}')
    # inh_legend = ax.scatter([], [], marker='|', color='red', label=f'inh | spikes/ms: {post_inh_rate:.2f}')
    # abl_legend = ax.scatter([], [], marker='|', color='black', label='ablated')
    # ax.legend(handles=[inh_legend, exc_legend, abl_legend], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.savefig(path + "/post_ablation_spike_raster", dpi = 300)

    ##### NOTE: Make everything pretty and plot it together #####
    fig, axs = plt.subplots(3, 4, figsize=(18,10))
    axs = axs.flatten()
    for ax in axs: ax.axis("off") # remove the markings from the original figure
    axs[0].imshow(plt.imread(path + "/IO_relationship" + ".png"), aspect='auto')
    axs[1].imshow(plt.imread(path + "/e_i_graph_output" + ".png"), aspect='auto')
    axs[2].imshow(plt.imread(path + "/i_e_graph_output" + ".png"), aspect='auto')
    axs[3].imshow(plt.imread(path + "/i_i_graph_output" + ".png"), aspect='auto')

    axs[4].imshow(plt.imread(path + "/pre_ablation_spike_raster" + ".png"), aspect='auto')
    axs[5].imshow(plt.imread(path + "/post_ablation_spike_raster" + ".png"), aspect = 'auto')
    axs[6].imshow(plt.imread(path + "/exc_state_space_rot_2" + ".png"), aspect='auto')
    axs[7].imshow(plt.imread(path + "/whole_state_space_rot_2" + ".png"), aspect='auto')

    axs[8].imshow(plt.imread(path + "/sv_plot" + ".png"), aspect='auto')
    axs[9].imshow(plt.imread(path + "/sv_cum_contribution" + ".png"), aspect='auto')
    axs[10].imshow(plt.imread(path + "/together_POV" + ".png"), aspect='auto')
    axs[11].imshow(plt.imread(path + "/pop_POV" + ".png"), aspect='auto')

    fig.suptitle(f'{name} -- params: {num_exc_neur} e neur || {num_inh_neur} i neur || e/i ratio {num_exc_neur/num_inh_neur} || e->i conn {e_i_precent_conn*100}% || i->e conn {i_e_precent_conn*100}% || i->i conn {i_i_precent_conn*100}% || pop ablation {ablation_frac*100}%', fontsize = 10)
    fig.savefig(path + "/overview_fig", dpi = 300)
    plt.tight_layout()

    print(f"\n\nlosers: {losers}\n\n")

    ##### NOTE: display/close functionality for plotting ease #####
    plt.show(block=False)
    plt.pause(0.001) # Pause for interval seconds.
    input("hit[enter] to end.")
    plt.close('all') # all open plots are correctly closed after each run