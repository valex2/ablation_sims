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
from nengo.solvers import LstsqL2
import nengo_bio as bio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

model = nengo.Network()

input1 = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
input2 = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# now make the two inputs columsn of a matrix
input = np.array([input1, input2]).T

# input = [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0],
#          [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
print(f"Input: {input}, input shape: {np.shape(input)}")

with model:
    stim = nengo.Node(PresentInput(input, 2))
    ensemble = nengo.Ensemble(100, dimensions=2)
    output = nengo.Node(size_in=2)

    nengo.Connection(stim, ensemble)
    nengo.Connection(ensemble, output)

    stim_probe = nengo.Probe(stim, synapse=0.01)
    ensemble_probe = nengo.Probe(ensemble, synapse=0.01)
    output_probe = nengo.Probe(output, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(40)

    plt.figure()
    plt.plot(sim.trange(), sim.data[stim_probe], label="Input")
    # plt.plot(sim.trange(), sim.data[ensemble_probe], label="Ensemble")
    # plt.plot(sim.trange(), sim.data[output_probe], label="Output")
    plt.legend()
    plt.show()