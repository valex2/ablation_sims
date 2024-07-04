#!/usr/bin/env python3

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

PROBE_SYNAPSE = 0.1 # Filter to be used for the network output
T = 10.0 # Total simulation time
T_SKIP = 1.0 # Time to exclude from the RMSE
SEED = 4891 # Network seed
SS = 100 # Plot subsample

def run_and_plot(model, probe, expected_fns, plot=True):
    # Run the simulation for the specified time
    with nengo.Simulator(model, progress_bar=None) as sim:
        sim.run(T)

    # Fetch the time and the probe data
    ts = sim.trange()
    expected = np.array([f(ts - PROBE_SYNAPSE) for f in expected_fns]).T
    actual = sim.data[probe]

    # Compute the slice over which to compute the error
    slice_ = slice(int(T_SKIP / sim.dt), int(T / sim.dt))

    # Compute the RMSE and the RMSE
    rms = np.sqrt(np.mean(np.square(expected)))
    rmse = np.sqrt(np.mean(np.square(expected[slice_] - actual[slice_])))

    if plot:
        fig, ax = plt.subplots()
        ax.plot(ts[::SS], expected[::SS], 'k--', label='Expected')
        ax.plot(ts[::SS], actual[::SS], label='Actual')
        ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.3))
        ax.set_title("Normalised RMSE = {:0.2f}%".format(100.0 * rmse / rms))
        ax.set_xlabel("Time $t$ (s)")
        ax.set_ylabel("Decoded value $x$")

    plt.show()

    return sim

def plot_weights(sim, conn):
    # Fetch the weights fro the model
    weights = sim.model.params[conn].weights
    WE, WI = weights[bio.Excitatory], weights[bio.Inhibitory]

    # Count the number of empty rows/columns
    def count_zero_rows(X):
        n = 0
        for i in range(X.shape[0]):
            if np.all(X[i] == 0):
                n += 1
        return n
    n_exc_zero_rows, n_inh_zero_rows = count_zero_rows(WE), count_zero_rows(WI)

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    axs[0].imshow( WE, vmin=0, vmax=np.median(WE[WE > 0]))
    axs[0].set_xlabel('Target neuron index')
    axs[0].set_ylabel('Source neuron index')
    axs[0].set_title('Excitatory ({} rows empty)'.format(n_exc_zero_rows))

    axs[1].imshow(-WI, vmin=0, vmax=np.median(-WI[WI < 0]))
    axs[1].set_xlabel('Target neuron index')
    axs[1].set_ylabel('Source neuron index')
    axs[1].set_title('Inhibitory ({} rows empty)'.format(n_inh_zero_rows))

    axs[2].imshow(WE-WI, vmin=0, vmax=np.median((WE-WI)[(WE-WI) > 0]))
    axs[2].set_xlabel('Target neuron index')
    axs[2].set_ylabel('Source neuron index')
    axs[2].set_title('Combined weights')

    fig.tight_layout()

    plt.show()

def connectivity_constraints_test(connectivity):
    with nengo.Network(seed=SEED) as model:
        inp_a = nengo.Node(lambda t: np.sin(t))

        ens_source = bio.Ensemble(n_neurons=101, dimensions=1, p_exc=0.8)
        ens_target = bio.Ensemble(n_neurons=103, dimensions=1)

        nengo.Connection(inp_a, ens_source)
        conn = bio.Connection(
            ens_source, ens_target,
            connectivity=connectivity
        )

        probe = nengo.Probe(ens_target, synapse=PROBE_SYNAPSE)

    sim = run_and_plot(model, probe, (lambda t: np.sin(t),), plot=False)
    plot_weights(sim, conn)

def main():
    f1, f2 = lambda t: np.sin(t), lambda t: np.cos(t)
    with nengo.Network(seed=SEED) as model:
        inp_a = nengo.Node(f1)
        inp_b = nengo.Node(f2)

        ens_a = bio.Ensemble(n_neurons=101, dimensions=1)
        ens_b = bio.Ensemble(n_neurons=102, dimensions=1)
        ens_c = bio.Ensemble(n_neurons=103, dimensions=2)

        nengo.Connection(inp_a, ens_a)
        nengo.Connection(inp_b, ens_b)

        bio.Connection((ens_a, ens_b), ens_c)

        probe = nengo.Probe(ens_c, synapse=PROBE_SYNAPSE)

    run_and_plot(model, probe, (f1, f2))
    connectivity_constraints_test(bio.ConstrainedConnectivity(convergence=10))

# This is the default connectivity that respects Dale's principle


if __name__ == '__main__':
  main()