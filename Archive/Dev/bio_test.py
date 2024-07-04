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
import nengo_bio as bio


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

num_exc_neur = 90
probe_synapse = 0.01
sim_duration = 10
ablation_frac = 0.5

with model: 
    ### population initialization ###
    joint_pop = bio.Ensemble(n_neurons = 100, dimensions = 1, p_exc = (num_exc_neur / 100))
    null_pop = bio.Ensemble(n_neurons = 100, dimensions = 1)

    input_signal = nengo.Node(output=layered_periodic_function)

    nengo.Connection(input_signal, joint_pop) #input to excitatory
    nengo.Connection(input_signal, null_pop) # connect to the null_pop
    
    ### probing ###
    e_probe = nengo.Probe(joint_pop, synapse=probe_synapse)
    null_probe = nengo.Probe(null_pop, synapse=probe_synapse)

    with nengo.Simulator(model) as sim:
        ##### NOTE: pre-ablation run #####
        sim.run(sim_duration)
        t = sim.trange()
        pre_ablation_e_probe = sim.data[e_probe][10:, :] # value at each milisecond

        ##### NOTE: null population metrics #####
        null_vals = sim.data[null_probe][10:,:] # null values

        #------------------------------------------------------------------------------------------------------------------------#
    
        losers = ablate_population([joint_pop], ablation_frac, sim)
        ablate_population([null_pop], ablation_frac, sim)

        ##### NOTE: post-ablation run #####
        sim.run(sim_duration)
        t = sim.trange()
        post_ablation_e_probe = sim.data[e_probe][sim_duration*1000 + 10:, :] # value at each milisecond
        post_ablation_null_probe = sim.data[null_probe][sim_duration*1000 + 10:, :]