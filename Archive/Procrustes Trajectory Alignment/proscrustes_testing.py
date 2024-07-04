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

frequency = 5  # Frequency in Hz
amplitude = 1  # Amplitude of the sinusoid
sampling_rate = 1000  # Sampling rate in Hz
duration = 1  # Duration in seconds
phase_shift = np.pi

t = np.linspace(0, duration, num=1000, endpoint=False) # Generate a time vector from 0 to 1 second with 1000 points

sinusoid = np.zeros((2, len(t))) # Create a 2D array to store the sinusoid
sinusoid[0, :] = np.transpose(amplitude * np.sin(2 * np.pi * frequency * t)) # Create the original sinusoidal signal
sinusoid[1, :] = t # Create the original sinusoidal signal

phase_shifted_sinusoid = np.zeros((2, len(t))) # Create a 2D array to store the sinusoid
phase_shifted_sinusoid[0,:] = np.transpose(amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)) # Create another sinusoid with a phase shift of π/2 (90 degrees)
phase_shifted_sinusoid[1,:] = t # Create another sinusoid with a phase shift of π/2 (90 degrees)

procrustes_similarity  = spatial.procrustes(sinusoid, phase_shifted_sinusoid)[2]

print(procrustes_similarity)