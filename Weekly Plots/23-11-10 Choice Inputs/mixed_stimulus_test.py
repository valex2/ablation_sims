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