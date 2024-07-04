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
import numpy as np
import nengo
from nengo.processes import WhiteSignal
from nengo.processes import PresentInput
from nengo.solvers import LstsqL2
from sklearn.preprocessing import StandardScaler
from nengo.utils.matplotlib import rasterplot
from sklearn.decomposition import PCA
import nengo_bio as bio

class recovery_module:
    def __init__(self, dimensions, timeperiod=1):
        self.history = np.zeros((timeperiod, dimensions))
        self.dimensions = dimensions

    ### simple stim

    ### PID control

    ### cost function can be computed from principal angles if a known 

    ### Laferriere

    ### STDP based learning rule
    # estimate manifold, compute new manifold
    # determine transformation of cov A to cov A'
    # implement clopath learning rule paper, 
    # LTP, LTD pairing for each


class neuron_ablation:

    """ 
    Default history timeperiod is just the last timestep, to enure 
    function for any selected simultation time. Select history timeperiod 
    for the ablation type of interest
    """

    def __init__(self, dimensions, timeperiod=1, ablate_fraction=0.2, ablate_rule='random', ablate_pc = 1):
        self.history = np.zeros((timeperiod, dimensions))
        self.dimensions = dimensions
        self.ablate_fraction = ablate_fraction
        self.ablate_rule = ablate_rule
        self.ablate_pc = 1
        self.current_clamp = np.min([-1000,-dimensions])

    def get_ablation_number(self):
        n_neurons_ablate = int(self.dimensions * self.ablate_fraction)
        return n_neurons_ablate

    def get_ablation_mask(self,t, x):
        """ 
        Note encoder weights are not zeroed out, currently. This function sets 
        the selected ablated neurons' bias current to min (-1000,self.dimensions) 
        preventing any output. Hyperpolarized voltage clamp set as -self.dimensions 
        to scale with network and prevent and voltage leak if simulating larger networks.
        """
        self.step(t, x)
        n_neurons_ablate = self.get_ablation_number()
        clamp = self.current_clamp

        if t == ablate_time:
            if self.ablate_rule == 'random':
                self.ablate_index = np.random.choice(np.arange(bnn.n_neurons), replace=False, size=n_neurons_ablate)
                ablate_mask = [clamp if i in self.ablate_index else 0 for i in range(len(x))]
                self.ablate_mask = ablate_mask
            elif self.ablate_rule == 'fr':
                frs = np.sum(self.history,axis=0)
                #self.probe=self.history
                self.ablate_index = np.argsort(frs)[-n_neurons_ablate:] # remove N largest
                ablate_mask = [clamp if i in self.ablate_index else 0 for i in range(len(x))]
                self.ablate_mask = ablate_mask
            elif self.ablate_rule == 'pca': 
                pc_num = self.ablate_pc
                binary = self.history.T
                print(np.shape(binary))
                t_bin = 25
                n_bin = int(len(binary[0,:])/t_bin)
                spike_counts = np.zeros((self.dimensions,n_bin))
                for j in range(n_bin) :  #-1 
                    spike_counts[:,j] = binary[:,j*t_bin:(j+1)*t_bin].sum(axis=1)
                X = spike_counts.transpose()
                X = StandardScaler().fit_transform(spike_counts)
                
                cov_mat = np.cov(X) #rowvar=False) #spike_counts)

                pca = PCA(n_components=self.dimensions)
                pca.fit(cov_mat)
                cov_trans = pca.transform(cov_mat)
                eig_vals = pca.singular_values_
                eig_vecs = pca.components_
                eig_sum = np.sum(np.abs(eig_vals))
                povs = np.sort(100*np.abs(eig_vals)/eig_sum)[::-1]
                I = np.eye(self.dimensions)
                weights = pca.transform(I)
                print(weights)
                print(povs)
                self.ablate_index = np.argsort(weights[self.ablate_pc-1,:])[-n_neurons_ablate:]
                ablate_mask = [clamp if i in self.ablate_index else 0 for i in range(len(x))]
                self.ablate_mask = ablate_mask
        elif t > ablate_time:
            ablate_mask = self.ablate_mask
        else:
            ablate_mask = [0 for i in range(len(x))]

        return ablate_mask

    def step(self, t, x):
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = x
        return self.history[0]

# def time_ablate(t):
#     # used for basic method, commented 
#     # out in model section
#     ablate_time = 10.0
#     return 1.0 if t > ablate_time else 0.0

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
    print(half1)
    print(half2)
    #source1 = np.random.binomial(1,0.5, size=(trial_num*(trial_duration+trial_interval),source_subsample)) 
    #source2 = np.random.binomial(1,0.5, size=(trial_num*(trial_duration+trial_interval),source_subsample))
    source1 = np.random.binomial(1,generator_probability, size=(trial_num,source_subsample)) 
    source2 = np.random.binomial(1,generator_probability, size=(trial_num,source_subsample))
    print(np.shape(source1))
    print(np.shape(source2))
    
    tsteps = int(trial_num*(trial_duration+trial_interval))
    mixture = np.zeros((trial_num*(trial_duration+trial_interval),source_number))
    counter_trial = 0  
    for t in range(tsteps):
        if t % int(trial_duration+trial_interval) == 0:
            
            if np.random.binomial(1,source_probability) == 1: 
                # s1 to h1
                mixture[t:t+trial_duration,half1] = source1[counter_trial]
            else:
                # s2 to h1
                mixture[t:t+trial_duration,half1] = source2[counter_trial]
            if np.random.binomial(1,source_probability) == 1:
                # s2 to h2
                mixture[t:t+trial_duration,half2] = source2[counter_trial]
            else:
                # s1 to h2
                mixture[t:t+trial_duration,half2] = source1[counter_trial]
            mixture[t+trial_duration:t+trial_duration+trial_interval,:] = np.zeros((source_number,))
            counter_trial += 1
    return mixture


## ----------------------------------------------------------------------------------------------- ##

### model parameters
dt = 0.001    # 1ms nengo default timestep, if not specified
sim_time = 20 # 20s of run time
n_timesteps = int(sim_time/dt)
ablate_fraction = 0.3
ablate_time = 10.0  # any integer multiple of the timestep 
n_neurons = 100
e_percent = 0.9 # TODO: make this a parameter
n_e_neurons = int(n_neurons * e_percent)
n_i_neurons = int(n_neurons * (1 - e_percent))

e_i_percent_conn = 0.05 # excitatory connectivity ratio
i_e_percent_conn = 0.05 # inhibitory connectivity ratio

n_sampled = 100
stim_dim = 2
ablate_rule = 'pca' 
ablate_pc = 1 

### history based ablation
ablation = neuron_ablation(n_neurons, timeperiod = int(ablate_time / dt), 
                          ablate_fraction = ablate_fraction, ablate_rule = ablate_rule, ablate_pc = ablate_pc)

ablation_2 = neuron_ablation(n_e_neurons, timeperiod = int(ablate_time / dt), 
                          ablate_fraction = ablate_fraction, ablate_rule = ablate_rule, ablate_pc = ablate_pc)

### generate stimulus sequence
source_number = n_neurons
source_sample = 32
trial_num = 256
trial_interval = 1000
trial_duration = 200
mixture_input = stochastic_mixture(trial_num=trial_num, trial_duration=trial_duration, 
                             trial_interval=trial_interval, source_number=source_number, 
                             source_sample=source_sample)

print(f'Mixture input: {mixture_input[1190:1210,:]}')
print(np.shape(mixture_input))

model = nengo.Network()
with model:
    ### model input node
    model_input = nengo.Node(WhiteSignal(n_neurons, high=5), size_out=2) #,size_in=2)
    
    # think about how to cluster neurons onto given inputs, one to many (few) mapping to neurons (e.g., 64 grid electrodes to 100 neurons) 
    model_input_test = nengo.Node(PresentInput(mixture_input,dt),size_out=n_neurons)
    bnn_test = nengo.Ensemble(n_neurons, dimensions=1)
    nengo.Connection(model_input_test,bnn_test.neurons)
    input_test_probe = nengo.Probe(model_input_test)
    bnn_test_probe = nengo.Probe(bnn_test, synapse=0.01)

    ### create and connect neuron ensembles
    bnn = nengo.Ensemble(n_neurons, dimensions=2)
    nengo.Connection(model_input, bnn)
    ann = nengo.Ensemble(n_neurons, dimensions=2)
    nengo.Connection(bnn, ann) 

    ### activity based bnn neuron ablation    
    ablate_node = nengo.Node(size_in=n_neurons, size_out=n_neurons, 
                             output=ablation.get_ablation_mask)
    nengo.Connection(bnn.neurons, ablate_node, synapse=None)
    nengo.Connection(ablate_node, bnn.neurons, synapse=0)

    ### define probes
    input_probe = nengo.Probe(model_input)
    bnn_probe = nengo.Probe(bnn, synapse=0.01)
    bnn_spikes = nengo.Probe(bnn.neurons)
    ann_probe = nengo.Probe(ann, synapse=0.01)
    #bnn_history_probe =  nengo.Probe(bnn_history)

    ### error meaure ensemble
    error = nengo.Ensemble(n_neurons, dimensions=2)
    error_probe = nengo.Probe(error, synapse=0.025)
    nengo.Connection(bnn, error) # actual
    nengo.Connection(model_input, error, transform=-1) # minus target

    #### ==================================================================================================== ####
    ### balanced network ###
    rand = np.random.randint(0, 4000) # seed the null and exc population such that they're the same for each run
    balanced_input = nengo.Node(WhiteSignal(n_neurons, high=5), size_out=1)
    e_pop = nengo.Ensemble(n_neurons = n_e_neurons, dimensions = 1, seed = rand)
    i_pop = nengo.Ensemble(n_neurons = n_i_neurons, dimensions = 1)
    null_pop = nengo.Ensemble(n_neurons = n_e_neurons, dimensions = 1, seed = rand)

    conn, e_e, e_i, i_e, i_i = balance_condition(n_e_neurons, n_i_neurons, e_i_percent_conn, i_e_percent_conn)

    nengo.Connection(balanced_input, e_pop) #input to excitatory
    nengo.Connection(e_pop.neurons, e_pop.neurons, transform = e_e)
    nengo.Connection(e_pop.neurons, i_pop.neurons, transform = e_i) # network connections
    nengo.Connection(i_pop.neurons, e_pop.neurons, transform = i_e)
    nengo.Connection(i_pop.neurons, i_pop.neurons, transform = i_i)
    nengo.Connection(balanced_input, null_pop) # connect to the null_pop

    ### error meaure ensemble
    balanced_error = nengo.Ensemble(n_neurons, dimensions=1)
    balanced_error_probe = nengo.Probe(balanced_error, synapse=0.025)
    nengo.Connection(e_pop, balanced_error) # actual
    nengo.Connection(balanced_input, balanced_error, transform=-1) # minus target

    ## probes ##
    e_probe = nengo.Probe(e_pop, synapse=0.01)
    e_spikes = nengo.Probe(e_pop.neurons)
    null_probe = nengo.Probe(null_pop, synapse=0.01)

    ### activity based bnn neuron ablation    
    ablate_node_2 = nengo.Node(size_in=n_e_neurons, size_out=n_e_neurons, 
                             output=ablation_2.get_ablation_mask)
    nengo.Connection(e_pop.neurons, ablate_node_2, synapse=0)
    nengo.Connection(ablate_node_2, e_pop.neurons, synapse=0)

    ### random neuron ablation in bnn ensemble - direct mask transformation (less general)
    #ablate = nengo.Node(time_ablate)
    #n_neurons_ablate = int(bnn.n_neurons * ablate_fraction)
    #ablate_index = np.random.choice(np.arange(bnn.n_neurons), replace=False, size=n_neurons_ablate)
    #ablate_mask = [[-1000] if i in ablate_index else [0] for i in range(bnn.n_neurons)]
    #nengo.Connection(ablate, bnn.neurons, transform = ablate_mask) 

    ### add learning rule to connection to try and correct (sort of works?)
    #conn.learning_rule_type = nengo.PES()
    # Connect the error into the learning rule
    #nengo.Connection(error, conn.learning_rule)

with nengo.Simulator(model) as sim:
    sim.run(sim_time)

plt.figure(figsize=(8,8))
plt.subplot(4, 1, 1)
plt.plot(sim.trange(), sim.data[input_probe].T[0], c="k", label="input")
plt.plot(sim.trange(), sim.data[bnn_probe].T[0], c="cornflowerblue", label="bnn")
#plt.plot(sim.trange(), sim.data[input_probe].T[1], c="k", label="Input")
#plt.plot(sim.trange(), sim.data[bnn_probe].T[1], c="cornflowerblue", label="Pre")
plt.legend(loc="best")

plt.subplot(4, 1, 2)
for i in range(n_neurons):
    plt.vlines(np.where(sim.data[bnn_spikes][:,i])[0]*dt,i,i+0.1,color='black')
# rasterplot(sim.trange(), sim.data[bnn_spikes])
plt.xlim(0,sim_time)

plt.subplot(4, 1, 3)
#plt.imshow(np.reshape(sim.data[input_test_probe].T,(n_neurons,n_neurons))) # change to these dimensions when deciding on 
#plt.plot(sim.trange(), np.sum(sim.data[input_test_probe].T,axis=0), c="k", label="input")
plt.plot(sim.trange(), sim.data[bnn_test_probe].T[0], c="darkred", label="ann")
#plt.plot(sim.trange(), sim.data[input_probe].T[1], c="k", label="Input")
#plt.plot(sim.trange(), sim.data[ann_probe].T[1], c="darkred", label="Pre")
plt.legend(loc="best")

plt.subplot(4, 1, 4)
plt.plot(sim.trange(), sim.data[error_probe].T[0], c="b",label="error")
plt.legend(loc="best")
plt.title("Bnn Model")
plt.ylim(-1, 1)

plt.figure(figsize=(8,8))
plt.subplot(4, 1, 1)
plt.plot(sim.trange(), sim.data[input_probe].T[0], c="k", label="input")
plt.plot(sim.trange(), sim.data[e_probe].T[0], c="cornflowerblue", label="exc")
plt.legend(loc="best")

plt.subplot(4, 1, 2)
for i in range(n_e_neurons):
    plt.vlines(np.where(sim.data[e_spikes][:,i])[0]*dt,i,i+0.1,color='black')
plt.xlim(0,sim_time)

plt.subplot(4, 1, 3)
plt.plot(sim.trange(), sim.data[null_probe].T[0], c="darkred", label="null")

plt.subplot(4, 1, 4)
plt.plot(sim.trange(), sim.data[balanced_error_probe].T[0], c="b",label="error")
plt.legend(loc="best")
plt.ylim(-1, 1)
plt.title("Original E-I Balance Model")
plt.show()

if __name__ == '__main__':
  main()