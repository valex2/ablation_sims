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
import csv
from tqdm import tqdm
# from bilm import bilm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
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
                    bias_sig[rel_neur_idx] = -1000000 # the bias term is set to 0, making the populations un-excitable
            
        group_idx = group_idx + ens.n_neurons
    return losers

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

def line_integral(data):
    squared_data = np.real(data) ** 2
    sum_across_features = np.sum(squared_data, axis=1)
    two_way_sum = np.log(np.sum(sum_across_features) / (len(data[0, :]) * len(data[:, 0])))
    return two_way_sum

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

##### NOTE: this function computes the mean square error between two sets of data (the data must have the same length)
def calculate_mse(baseline, for_comparison):
    if len(for_comparison) != len(baseline):
        raise ValueError("Signals must have the same length")
    
    squared_diff = np.square(for_comparison - baseline)
    mse = np.mean(squared_diff)
    return mse

model = nengo.Network(label="E/I offset")

##### NOTE: CREATE SIMULATION FOLDER AND CONFIG #####
os.environ
RUNNER = os.environ['RUNNER']
ARRAY_ID = int(os.environ['NUM_JOB']) # ARRAY ID CAN BE USED TO INDEX INTO A VARIABLE OF INTEREST --> param = param_range[ARRAY_ID]

e_i_ratio_range = np.linspace(0.99, 0.01, 50) # TODO this is the parameter space that we range across
rel_e_i = e_i_ratio_range[ARRAY_ID] # pull this based off of the run ID
NUM_TOTAL = 1000 # number of total neurons for rel e:i
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
              'num_exc_neur' : num_exc,
              'num_inh_neur' : num_inh,
              'ablation_frac' : 0.7, # fraction of neurons to lesion
              
              'e_i_precent_conn' : 0.05, # excitatory connectivity ratio
              'e_i_max_val' : 0.1, # the greatest strength of the e_i connection
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
ablation_frac = slurm_dict['ablation_frac']
time_cut_off = slurm_dict['time_cut_off']
lesion_bars = slurm_dict['lesion_bars']

with model: 
    ### population initialization ###
    rand_num = np.random.randint(0, 4000) # seed the null and exc population such that they're the same
    exc_pop = nengo.Ensemble(n_neurons = num_exc_neur, dimensions = 1, seed = rand_num)
    inh_pop = nengo.Ensemble(n_neurons = num_inh_neur, dimensions = 1)
    null_pop = nengo.Ensemble(n_neurons = num_exc_neur, dimensions = 1, seed = rand_num) # baseline to see whether signal is reproducible given population size
    input_signal = nengo.Node(output=layered_periodic_function)
    
    ### connectivity ###
    # NOTE: a transform is a linear transformation  mapping the pre function output to the post function input
    inhibitory_transform_i_e = sparse_connectivity(i_e_precent_conn, i_e_max_val, inh_pop.n_neurons, exc_pop.n_neurons, i_e_tailed_dist) # -1 since inhibitory
    inhibitory_transform_i_i = sparse_connectivity(i_i_precent_conn, i_i_max_val, inh_pop.n_neurons, inh_pop.n_neurons, i_i_tailed_dist) # -1 since inhibitory
    excitatory_transform = sparse_connectivity(e_i_precent_conn, e_i_max_val, exc_pop.n_neurons, inh_pop.n_neurons, e_i_tailed_dist)
    nengo.Connection(input_signal, exc_pop) #input to excitatory
    nengo.Connection(exc_pop.neurons, inh_pop.neurons, transform = excitatory_transform) # network connections
    nengo.Connection(inh_pop.neurons, exc_pop.neurons, transform = inhibitory_transform_i_e)
    nengo.Connection(inh_pop.neurons, inh_pop.neurons, transform = inhibitory_transform_i_i)
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

    # ##### NOTE: joint population metrics, pre-ablation #####
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

    # ##### NOTE: SVD power plots, null #####
    # U, S, VT = np.linalg.svd(Y, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    # S_null = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else

    # ##### NOTE: SVD power plots, pre-ablation #####
    # U, S, VT = np.linalg.svd(X, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    # S_pre_ablation = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else

    # ##### NOTE: SVD power plots, pre-ablation, joint #####
    # U, S, VT = np.linalg.svd(T, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    # S_joint_pre_ablation = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else

    # #------------------------------------------------------------------------------------------------------------------------#
   
    ##### NOTE: performing ablation #####
    ensembles = []
    if lesion_e:
        ensembles.append(exc_pop)
    if lesion_i:
        ensembles.append(inh_pop)

    losers = ablate_population(ensembles, ablation_frac, sim)

    ##### NOTE: post-ablation run #####
    sim.run(sim_duration)
    t = sim.trange()
    print(f"\nsim duration: {sim_duration}, t (sim_range) = {t}\n")
    e_spk = sim.data[e_spikes][sim_duration*1000 + 10:, :] # spiking at each milisecond -- (length of sim (ms) x num_neurons)
    post_ablation_e_probe = sim.data[e_probe][sim_duration*1000 + 10:, :] # value at each milisecond
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
    ##### NOTE: saving files #####

    povs_path = os.path.join(path, 'povs')  # Main directory

    if not os.path.exists(povs_path):
        os.makedirs(povs_path)

    subdirectories = [
        ('povs_pre', povs_pre),
        ('povs_joint_pre', joint_povs_pre),
        ('povs_null', null_povs),
        ('povs_post', povs_post),
        ('povs_joint_post', joint_povs_post)
    ]
    
    for subdirectory_name, data in subdirectories:
        subdirectory_path = os.path.join(povs_path, subdirectory_name)
        if not os.path.exists(subdirectory_path):
            os.makedirs(subdirectory_path)

        file_path = os.path.join(subdirectory_path, f"job_{ARRAY_ID}_E:I_{rel_e_i:2f}_{subdirectory_name}.csv")
        np.savetxt(file_path, data, delimiter=',', fmt='%1.3f')

    # #------------------------------------------------------------------------------------------------------------------------#

    # ##### NOTE: state space plot initialization, exc pop #####
    # exc_state_space = plt.figure() # Create a 3D plot
    # ax = exc_state_space.add_subplot(111, projection='3d')
    # ax.set_xlabel('PC 1', labelpad=-2) # Set labels for the axes
    # ax.set_ylabel('PC 2', labelpad=-2)
    # ax.set_zlabel('PC 3', labelpad=-2)
    # ax.set_xticks([]) # remove the tick marks from the axis
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.set_title("pre/post ablation state space, exc pop")
    # cmap = plt.get_cmap('Blues')

    # ##### NOTE: state space plotting, pre-ablation #####
    idx = pre_eig_vals.argsort()[::-1] # sort the eigenvalues and eigenvectors from highest to lowest  
    pre_eig_vals = pre_eig_vals[idx] # eigval shape is (40,)
    pre_eig_vect = pre_eig_vect[:,idx] # eigvector shape is (40 x 40) where each column represents an eigenvector
    eig3D = pre_eig_vect[:, :3]
    pre_ab_neural_state_space = pre_e_spike_count.T @ eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space # TODO save this
    # num_time_steps = neural_state_space.shape[0] # Create a colormap based on the number of time steps
    # for i in range(num_time_steps): # plot the points with gradient coloring
    #     x = neural_state_space[i, 0]
    #     y = neural_state_space[i, 1]
    #     z = neural_state_space[i, 2]
    #     ax.scatter(x, y, z, c=[cmap(i / num_time_steps)], s=10)  # Adjust s for point size

    # ##### NOTE: plane plotting #####
    fit, residual = plane_of_best_fit(pre_ab_neural_state_space)
    xmin, xmax = np.min(pre_ab_neural_state_space[:, 0]), np.max(pre_ab_neural_state_space[:, 0]) # specify linspace overwhich to plot the plane based on datapoints
    ymin, ymax = np.min(pre_ab_neural_state_space[:, 1]), np.max(pre_ab_neural_state_space[:, 1])
    xx, yy = np.meshgrid(np.linspace(xmin, xmax), np.linspace(ymin, ymax)) # define the grid
    pre_z = (fit[0] * xx + fit[1] * yy + fit[2])
    pre_norm = (-fit[0], -fit[1], 1)
    # ax.plot_surface(xx, yy, pre_z, alpha = 0.3, color = "lightblue") # plot the plane
    pre_ab_marker = line_integral(pre_ab_neural_state_space)
    # ax.plot(neural_state_space[0:(num_time_steps - 1),0], neural_state_space[0:(num_time_steps - 1),1], neural_state_space[0:(num_time_steps - 1),2], c = "blue", label = f"pre-ablation trajectory, λ={marker:.2f}", alpha = 0.4) # plot the trajectory

    # ##### NOTE: state space plotting, post-ablation #####
    # ### -- we're projecting the ablated data into the same eigenspace as the healthy data for comparison purposes -- ###
    post_ab_neural_state_space = post_e_spike_count.T @ eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space # TODO save this
    # cmap = plt.get_cmap('inferno')
    # for i in range(num_time_steps): # plot the points with gradient coloring
    #     x = neural_state_space[i, 0]
    #     y = neural_state_space[i, 1]
    #     z = neural_state_space[i, 2]
    #     ax.scatter(x, y, z, c=[cmap(i / num_time_steps)], s=10)  # Adjust s for point size
    post_ab_marker = line_integral(post_ab_neural_state_space)
    
    # ##### NOTE: plane plotting #####
    fit, residual = plane_of_best_fit(post_ab_neural_state_space)
    post_z = (fit[0] * xx + fit[1] * yy + fit[2])
    # ax.plot_surface(xx, yy, post_z, alpha = 0.3, color = "lightcoral") # plot the plane
    post_norm = (-fit[0], -fit[1], 1)

    # ##### NOTE: compute the difference in angle
    del_angle_radians = np.arccos(np.dot(pre_norm, post_norm) / (np.linalg.norm(pre_norm) * np.linalg.norm(post_norm)))
    del_angle_degrees = [np.degrees(np.real(del_angle_radians))[0]]
    # ax.plot(neural_state_space[0:(num_time_steps - 1),0], neural_state_space[0:(num_time_steps - 1),1], neural_state_space[0:(num_time_steps - 1),2], c = "red", label = f"post-ablation trajectory, λ={marker:.2f}, θ={del_angle_degrees:.2f}º", alpha = 0.4) # plot the trajectory

    plane_path = os.path.join(path, 'exc_plane')  # Main directory
    subdirectories = [
        ('pre_ab_neural_state_space', pre_ab_neural_state_space),
        ('post_ab_neural_state_space', post_ab_neural_state_space)]
    #     ('pre_marker', pre_ab_marker),
    #     ('post_marker', post_ab_marker),

    if not os.path.exists(plane_path):
        os.makedirs(plane_path)

    for subdirectory_name, data in subdirectories:
        subdirectory_path = os.path.join(plane_path, subdirectory_name)
        if not os.path.exists(subdirectory_path):
            os.makedirs(subdirectory_path)

        file_path = os.path.join(subdirectory_path, f"job_{ARRAY_ID}_E:I_{rel_e_i:2f}_{subdirectory_name}.csv")
        np.savetxt(file_path, data, delimiter=',', fmt='%1.3f')

    subdirectory_path = os.path.join(plane_path, 'degree')
    if not os.path.exists(subdirectory_path):
            os.makedirs(subdirectory_path)
    file_path = os.path.join(subdirectory_path, f"job_{ARRAY_ID}_E:I_{rel_e_i:2f}_{subdirectory_name}.csv")
    np.savetxt(file_path, data, delimiter=',', fmt='%1.3f')

    ##### NOTE: MSE calculation #####
    baseline = layered_periodic_function((t[10:(sim_duration * 1000)]))
    mse_to_null = calculate_mse(null_vals, baseline)
    mse_to_pre = calculate_mse(pre_ablation_e_probe, baseline)
    mse_to_post = calculate_mse(post_ablation_e_probe, baseline)

    data_dict = {  # Create a dictionary
        'Condition': ['Null', 'Pre-Ablation', 'Post-Ablation'],
        'MSE to Baseline': [mse_to_null, mse_to_pre, mse_to_post]
    }

    mse_path = os.path.join(path, 'mse')  # Main directory
    if not os.path.exists(mse_path):
        os.makedirs(mse_path)

    csv_filename = f"job_{ARRAY_ID}_E:I_{rel_e_i:2f}_mse.csv"
    csv_path = os.path.join(mse_path, csv_filename)
    
    with open(csv_path, 'w', newline='') as csv_file: # Save the dictionary data as CSV
        writer = csv.writer(csv_file)
        writer.writerow(data_dict.keys()) # Write the header row
        
        for row_data in zip(*data_dict.values()): # Write the data rows
            writer.writerow(row_data)

    print(f'Data saved to {csv_path}')

    # #------------------------------------------------------------------------------------------------------------------------#

    # ##### NOTE: state space plot initialization, whole pop #####
    # whole_state_space = plt.figure() # Create a 3D plot
    # ax = whole_state_space.add_subplot(111, projection='3d')
    # ax.set_xlabel('PC 1', labelpad=-2) # Set labels for the axes
    # ax.set_ylabel('PC 3', labelpad=-2)
    # ax.set_zlabel('PC 2', labelpad=-2)
    # ax.set_xticks([]) # remove the tick marks from the axis
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.set_title("pre/post ablation state space, whole pop")
    # cmap = plt.get_cmap('Blues')

    # ##### NOTE: state space plotting, pre-ablation, whole pop #####
    # idx = pre_joint_eig_vals.argsort()[::-1] # sort the eigenvalues and eigenvectors from highest to lowest  
    # pre_joint_eig_vals = pre_joint_eig_vals[idx] # eigval shape is (40,)
    # pre_joint_eig_vect = pre_joint_eig_vect[:,idx] # eigvector shape is (40 x 40) where each column represents an eigenvector
    # joint_eig3D = pre_joint_eig_vect[:, :3]
    # neural_state_space = pre_joint_spike_count.T @ joint_eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space
    # neural_state_space = np.real(neural_state_space) ## TODO
    # reorder = [0, 2, 1]
    # neural_state_space = neural_state_space[:, reorder]
    # num_time_steps = neural_state_space.shape[0] # Create a colormap based on the number of time steps
    # for i in range(num_time_steps): # plot the points with gradient coloring
    #     x = neural_state_space[i, 0]
    #     y = neural_state_space[i, 1]
    #     z = neural_state_space[i, 2]
    #     ax.scatter(x, y, z, c=[cmap(i / num_time_steps)], s=10)  # Adjust s for point size
    # marker = line_integral(neural_state_space)
    # ax.plot(neural_state_space[0:(num_time_steps - 1),0], neural_state_space[0:(num_time_steps - 1),1], neural_state_space[0:(num_time_steps - 1),2], c = "blue", label = f"pre-ablation trajectory, λ={marker:.3f}", alpha = 0.4) # plot the trajectory

    # ##### NOTE: plane plotting #####
    # fit, residual = plane_of_best_fit(neural_state_space)
    # xmin, xmax = np.min(neural_state_space[:, 0]), np.max(neural_state_space[:, 0]) # specify linspace overwhich to plot the plane based on datapoints
    # ymin, ymax = np.min(neural_state_space[:, 1]), np.max(neural_state_space[:, 1])
    # xx, yy = np.meshgrid(np.linspace(xmin, xmax), np.linspace(ymin, ymax)) # define the grid
    # pre_z = (fit[0] * xx + fit[1] * yy + fit[2])
    # pre_norm = (-fit[0], -fit[1], 1)
    # ax.plot_surface(xx, yy, pre_z, alpha = 0.3, color = "lightblue") # plot the plane

    # ##### NOTE: state space plotting, post-ablation, whole pop #####
    # neural_state_space = post_joint_spike_count.T @ joint_eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space
    # neural_state_space = np.real(neural_state_space) ## TODO
    # neural_state_space = neural_state_space[:, reorder]
    # num_time_steps = neural_state_space.shape[0] # Create a colormap based on the number of time steps
    # cmap = plt.get_cmap('inferno')
    # for i in range(num_time_steps): # plot the points with gradient coloring
    #     x = neural_state_space[i, 0]
    #     y = neural_state_space[i, 1]
    #     z = neural_state_space[i, 2]
    #     ax.scatter(x, y, z, c=[cmap(i / num_time_steps)], s=10)  # Adjust s for point size

    # ##### NOTE: plane plotting #####
    # fit, residual = plane_of_best_fit(neural_state_space)
    # post_z = (fit[0] * xx + fit[1] * yy + fit[2])
    # post_norm = (-fit[0], -fit[1], 1)
    # ax.plot_surface(xx, yy, post_z, alpha = 0.3, color = "lightcoral") # plot the plane

    # ##### NOTE: plane metrics #####
    # joint_del_angle_radians = np.arccos(np.dot(pre_norm, post_norm) / (np.linalg.norm(pre_norm) * np.linalg.norm(post_norm)))
    # joint_del_angle_degrees = np.degrees(joint_del_angle_radians)[0]
    # marker = line_integral(neural_state_space)
    # ax.plot(neural_state_space[0:(num_time_steps - 1),0], neural_state_space[0:(num_time_steps - 1),1], neural_state_space[0:(num_time_steps - 1),2], c = "red", label = f"post-ablation trajectory, λ={marker:.2f}, θ={joint_del_angle_degrees:.2f}", alpha = 0.4) # plot the trajectory

    # ##### NOTE: making the state space plot look nice #####
    # plt.legend()
    # ax.view_init(elev=33, azim=11, roll = 0) # rotate the plot
    # whole_state_space.savefig(path + "/whole_state_space_rot_1", dpi = 300)
    # ax.view_init(elev=46, azim=138, roll = 0) # rotate the plot
    # whole_state_space.savefig(path + "/whole_state_space_rot_2", dpi = 300)

    # #------------------------------------------------------------------------------------------------------------------------#

    # ##### NOTE: state space plot initialization, null pop #####
    # null_state_space = plt.figure() # Create a 3D plot
    # ax = null_state_space.add_subplot(111, projection='3d')
    # ax.set_xlabel('PC 1', labelpad=-2) # Set labels for the axes
    # ax.set_ylabel('PC 2', labelpad=-2)
    # ax.set_zlabel('PC 3', labelpad=-2)
    # ax.set_xticks([]) # remove the tick marks from the axis
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.set_title("null pop state space")
    # cmap = plt.get_cmap('Greens')

    # ##### NOTE: state space plotting null pop #####
    # idx = null_eig_vals.argsort()[::-1] # sort the eigenvalues and eigenvectors from highest to lowest  
    # null_eig_vals = null_eig_vals[idx] # eigval shape is (40,)
    # null_eig_vect = null_eig_vect[:,idx] # eigvector shape is (40 x 40) where each column represents an eigenvector
    # null_eig3D = null_eig_vect[:, :3]
    # neural_state_space = null_spike_count.T @ null_eig3D # project into the PCA space -- first arg is 200x40 - (200 x 40) x (40 x 3) yields 200 points in 3 space
    # num_time_steps = neural_state_space.shape[0] # Create a colormap based on the number of time steps
    # for i in range(num_time_steps): # plot the points with gradient coloring
    #     x = neural_state_space[i, 0]
    #     y = neural_state_space[i, 1]
    #     z = neural_state_space[i, 2]
    #     ax.scatter(x, y, z, c=[cmap(i / num_time_steps)], s=10)  # Adjust s for point size
    # ax.plot(neural_state_space[0:(num_time_steps - 1),0], neural_state_space[0:(num_time_steps - 1),1], neural_state_space[0:(num_time_steps - 1),2], c = "green", label = "null trajectory", alpha = 0.4) # plo

    # ##### NOTE: making the state space plot look nice #####
    # plt.legend()
    # ax.view_init(elev=33, azim=11, roll = 0) # rotate the plot
    # null_state_space.savefig(path + "/null_state_space_rot_1", dpi = 300)
    # ax.view_init(elev=46, azim=138, roll = 0) # rotate the plot
    # null_state_space.savefig(path + "/null_state_space_rot_2", dpi = 300)

    # #------------------------------------------------------------------------------------------------------------------------#

    # ##### NOTE: SVD power plots, post-ablation #####
    # U, S, VT = np.linalg.svd(X, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    # S_post_ablation = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else

    # ##### NOTE: SVD power plots, post-ablation, joint #####
    # U, S, VT = np.linalg.svd(T, full_matrices=False) # output singular value decomposition of X, the standardized spike rate matrix
    # S_joint_post_ablation = np.diag(S) # diagonalize the S matrix. From (n_neuron x 1) -- to (n_neuron x n_neuron) with 0s everywhere else
    
    # ##### NOTE: POV plotting, excitatory population only #####
    # together_POV, together_var = plot_POVs([null_povs, povs_pre, povs_post], [f"null", f"pre-ablation, {i_e_precent_conn*100}% connected", f"post-ablation, {i_e_precent_conn*100}% connected"], ["lightgreen", "lightblue", "lightcoral"], f"excitatory pop. variance explained by top {num_eigsum} PCs")
    # together_POV.savefig(path + "/together_POV", dpi = 300)

    # ##### NOTE: POV plotting, entire population #####
    # pop_POV, pop_var = plot_POVs([joint_povs_pre, joint_povs_post], [f"pre-ablation, {i_e_precent_conn*100}% connected", f"post-ablation, {i_e_precent_conn*100}% connected"], ["lightblue", "lightcoral"], f"whole pop. variance explained by top {num_eigsum} PCs")
    # pop_POV.savefig(path + "/pop_POV", dpi = 300)
    
    # ##### NOTE: inter-connection connectivity plotting #####

    # ##### NOTE: SV/power plotting
    # sv_plot = plt.figure() # singular value plotting
    # plt.semilogy(np.diag(S_null), linestyle = "dashed", label="null", color = "lightgreen")
    # plt.semilogy(np.diag(S_pre_ablation), linestyle = "dashed", label="exc pop, pre-ablation", color = "lightblue")
    # plt.semilogy(np.diag(S_post_ablation), linestyle = "dashed", label="exc pop, post-ablation", color = "lightcoral")
    # plt.semilogy(np.diag(S_joint_pre_ablation), linestyle = "solid", label="whole pop, pre-ablation", color = "lightblue")
    # plt.semilogy(np.diag(S_joint_post_ablation), linestyle = "solid", label="whole pop, post-ablation", color = "lightcoral")
    # plt.title('SV Log-Scale Power')
    # plt.xlabel("singular value #")
    # plt.ylabel("power of singular value")
    # plt.legend()
    # sv_plot.savefig(path + "/sv_plot", dpi = 300)

    # sv_cum_contribution = plt.figure()
    # plt.plot(np.cumsum((np.diag(S_null))/np.sum(np.diag(S_null)))*100, linestyle = "dashed", label = "null", color = "lightgreen")
    # plt.plot(np.cumsum((np.diag(S_pre_ablation))/np.sum(np.diag(S_pre_ablation)))*100, linestyle = "dashed", label = "exc pop, pre-ablation", color = "lightblue")
    # plt.plot(np.cumsum((np.diag(S_post_ablation))/np.sum(np.diag(S_post_ablation)))*100, linestyle = "dashed", label = "exc pop, post-ablation", color = "lightcoral")
    # plt.plot(np.cumsum((np.diag(S_joint_pre_ablation))/np.sum(np.diag(S_joint_pre_ablation)))*100, linestyle = "solid", label = "whole pop, pre-ablation", color = "lightblue")
    # plt.plot(np.cumsum((np.diag(S_joint_post_ablation))/np.sum(np.diag(S_joint_post_ablation)))*100, linestyle = "solid", label = "whole pop, post-ablation", color = "lightcoral")
    # plt.title('SV Cumulative Contribution')
    # plt.xlabel("# of singular values")
    # plt.ylabel("percent contribution to overall data")
    # plt.legend()
    # sv_cum_contribution.savefig(path + "/sv_cum_contribution", dpi = 300)

    # ##### NOTE: e-i connectivity plotting #####
    # plt.figure()
    # max = np.abs(e_i_max_val)
    # sns.heatmap(excitatory_transform, annot=False, vmin = -max, vmax = max, cmap = cm.RdBu)
    # plt.ylabel("inhibitory population neurons")
    # plt.xlabel("excitatory population neurons")
    # plt.title("e->i Connectivity Map")
    # plt.savefig(path + "/e_i_graph_output", dpi = 300)

    # ##### NOTE: intra-connection connectivity plotting #####
    # plt.figure()
    # max = np.abs(i_e_max_val)
    # sns.heatmap(inhibitory_transform_i_e, annot=False, vmin = -max, vmax = max, cmap = cm.RdBu)
    # plt.ylabel("excitatory population neurons")
    # plt.xlabel("inhibitory population neurons")
    # plt.title("i->e Connectivity Map")
    # plt.savefig(path + "/i_e_graph_output", dpi = 300)

    # ##### NOTE: recurrent-connection connectivity plotting #####
    # plt.figure()
    # max = np.abs(i_i_max_val)
    # sns.heatmap(inhibitory_transform_i_i, annot=False, vmin = -max, vmax = max, cmap = cm.RdBu)
    # plt.ylabel("inhibitory population neurons")
    # plt.xlabel("inhibitory population neurons")
    # plt.title("i->i Connectivity Map")
    # plt.savefig(path + "/i_i_graph_output", dpi = 300)

    # ##### NOTE: output plotting #####
    # plt.figure()
    # plt.plot(t[10:(sim_duration * 1000)], layered_periodic_function(t[10:(sim_duration * 1000)]), label = f"input", color = "green")
    # plt.plot(t[10:(sim_duration * 1000)], null_vals, label = f"unconnected output", color = "blue")
    # plt.plot(t[10:(sim_duration * 1000)], pre_ablation_e_probe, label = f"{i_e_precent_conn*100}% connected output", color = "orange")
    # plt.plot(t[10:(sim_duration * 1000)], post_ablation_e_probe, label = f"{i_e_precent_conn*100}% connected, {ablation_frac*100}% ablated output", color = "red")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.title("I/O Relationship")
    # plt.legend()
    # plt.savefig(path + "/IO_relationship", dpi = 300)

    

    # ##### NOTE: pre-ablation raster plotting #####
    # fig, ax = plt.subplots()
    # pre_joint_binary = pre_joint_binary[:, :time_cut_off]
    # pre_spike_rates = np.sum(pre_joint_binary, axis=1, keepdims=True) # sum across the time domain
    # pre_exc_rate = np.sum(pre_spike_rates[:num_exc_neur]) / (time_cut_off * num_exc_neur) # average across time
    # pre_inh_rate = np.sum(pre_spike_rates[num_exc_neur : num_exc_neur+num_inh_neur]) / (time_cut_off * num_inh_neur) # average across time
    # for neuron_idx in range(num_exc_neur + num_inh_neur):
    #     spike_times = np.where(pre_joint_binary[neuron_idx])[0]
    #     ax.scatter(spike_times, np.full_like(spike_times, neuron_idx), marker='|', color='green' if neuron_idx < num_exc_neur else 'red', s=0.5)
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Neuron')
    # ax.set_title('spike raster plot, pre-ablation')
    # ax.axhspan(0, num_exc_neur, facecolor='green', alpha=0.05, label='excitatory neurons')
    # ax.axhspan(num_exc_neur, num_exc_neur + num_inh_neur, facecolor='red', alpha=0.05, label='inhibitory neurons')
    # ax.set_ylim(-1, num_exc_neur + num_inh_neur)
    # exc_legend = ax.scatter([], [], marker='|', color='g', label=f'exc | spikes/ms: {pre_exc_rate:.2f}')
    # inh_legend = ax.scatter([], [], marker='|', color='red', label=f'inh | spikes/ms: {pre_inh_rate:.2f}')
    # ax.legend(handles=[inh_legend, exc_legend], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    # plt.savefig(path + "/pre_ablation_spike_raster", dpi = 300)

    # ##### NOTE: post-ablation raster plotting #####
    # fig, ax = plt.subplots()
    # post_joint_binary = post_joint_binary[:, :time_cut_off]
    # post_spike_rates = np.sum(post_joint_binary, axis=1, keepdims=True) # sum across the time domain
    # post_exc_rate = np.sum(post_spike_rates[:num_exc_neur]) / (time_cut_off * num_exc_neur) # average across time
    # post_inh_rate = np.sum(post_spike_rates[num_exc_neur : num_exc_neur+num_inh_neur]) / (time_cut_off * num_inh_neur) # average across time
    # for neuron_idx in range(num_exc_neur + num_inh_neur):
    #     spike_times = np.where(post_joint_binary[neuron_idx])[0]
    #     ax.scatter(spike_times, np.full_like(spike_times, neuron_idx), marker='|', color='g' if neuron_idx < num_exc_neur else 'red', s=0.5)
    # if lesion_bars:
    #     for loser_y in losers: # show which neurons were ablated
    #         ax.axhline(y=loser_y, color='black', linewidth=1, alpha = 0.5) 
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Neuron')
    # ax.set_title('spike raster plot, post-ablation')
    # ax.axhspan(0, num_exc_neur, facecolor='green', alpha=0.05, label='excitatory neurons')
    # ax.axhspan(num_exc_neur, num_exc_neur + num_inh_neur, facecolor='red', alpha=0.05, label='inhibitory neurons')
    # ax.set_ylim(-1, num_exc_neur + num_inh_neur)
    # exc_legend = ax.scatter([], [], marker='|', color='g', label=f'exc | spikes/ms: {post_exc_rate:.2f}')
    # inh_legend = ax.scatter([], [], marker='|', color='red', label=f'inh | spikes/ms: {post_inh_rate:.2f}')
    # abl_legend = ax.scatter([], [], marker='|', color='black', label='ablated')
    # ax.legend(handles=[inh_legend, exc_legend, abl_legend], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    # plt.savefig(path + "/post_ablation_spike_raster", dpi = 300)

    # ##### NOTE: Make everything pretty and plot it together #####
    # fig, axs = plt.subplots(3, 4, figsize=(18,10))
    # axs = axs.flatten()
    # for ax in axs: ax.axis("off") # remove the markings from the original figure
    # axs[0].imshow(plt.imread(path + "/IO_relationship" + ".png"), aspect='auto')
    # axs[1].imshow(plt.imread(path + "/e_i_graph_output" + ".png"), aspect='auto')
    # axs[2].imshow(plt.imread(path + "/i_e_graph_output" + ".png"), aspect='auto')
    # axs[3].imshow(plt.imread(path + "/i_i_graph_output" + ".png"), aspect='auto')

    # axs[4].imshow(plt.imread(path + "/pre_ablation_spike_raster" + ".png"), aspect='auto')
    # axs[5].imshow(plt.imread(path + "/post_ablation_spike_raster" + ".png"), aspect = 'auto')
    # axs[6].imshow(plt.imread(path + "/exc_state_space_rot_2" + ".png"), aspect='auto')
    # axs[7].imshow(plt.imread(path + "/whole_state_space_rot_2" + ".png"), aspect='auto')

    # axs[8].imshow(plt.imread(path + "/sv_plot" + ".png"), aspect='auto')
    # axs[9].imshow(plt.imread(path + "/sv_cum_contribution" + ".png"), aspect='auto')
    # axs[10].imshow(plt.imread(path + "/together_POV" + ".png"), aspect='auto')
    # axs[11].imshow(plt.imread(path + "/pop_POV" + ".png"), aspect='auto')

    # fig.suptitle(f'Params: {num_exc_neur} e neur || {num_inh_neur} i neur || e/i ratio {num_exc_neur/num_inh_neur} || e->i conn {e_i_precent_conn*100}% || i->e conn {i_e_precent_conn*100}% || i->i conn {i_i_precent_conn*100}% || pop ablation {ablation_frac*100}%', fontsize = 10)
    # fig.savefig(path + "/overview_fig", dpi = 300)
    # plt.tight_layout()

    # print(f"\n\nlosers: {losers}\n\n")

    # ##### NOTE: display/close functionality for plotting ease #####
    # plt.show(block=False)
    # plt.pause(0.001) # Pause for interval seconds.
    # input("hit[enter] to end.")
    # plt.close('all') # all open plots are correctly closed after each run