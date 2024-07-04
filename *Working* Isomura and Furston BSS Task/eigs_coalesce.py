import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

os.environ
# OUT_DIR = "/Users/Vassilis/Desktop/BIL/Winter 24 Writeup/Images from Data" #NOTE: change this for output folder
OUT_DIR = "/Users/Vassilis/Desktop/BIL/Winter 24 Writeup/Images from Data/4x64_connected"
os.system('mkdir {}'.format(os.path.join(OUT_DIR,'plots')))

# azimuth = -95, elevation= 41

name = "4x64 connected" # NOTE: change this for the name of the experiment
accuracy = False # NOTE: change this for the type of experiment

# data_path = f"/Users/Vassilis/Desktop/BIL/Winter 24 Writeup/Data/Continuous_tracking"
data_path = f"/Users/Vassilis/Desktop/BIL/Winter 24 Writeup/Data/4x64_weighted_eigs"
path = os.path.join(OUT_DIR,f'{name}')
if not os.path.exists(path):
        os.makedirs(path)

e_to_i_range = np.arange(0.1, 0.91, 0.01) # 81 different parameter values
num_files = len(e_to_i_range)

frobenius_range = 200 # number of points to sample in the frobenius distance range
num_iter = 21 #30
num_samples = num_iter #int((num_iter *(num_iter - 1))/2)
num_sources = 4

pre_eigs1 = np.zeros((num_files, num_samples))
pre_eigs2 = np.zeros((num_files, num_samples))
pre_eigs3 = np.zeros((num_files, num_samples))

post_eigs1 = np.zeros((num_files, num_samples))
post_eigs2 = np.zeros((num_files, num_samples))
post_eigs3 = np.zeros((num_files, num_samples))


pre_joint_eigs1 = np.zeros((num_files, num_samples))
pre_joint_eigs2 = np.zeros((num_files, num_samples))
pre_joint_eigs3 = np.zeros((num_files, num_samples))

post_joint_eigs1 = np.zeros((num_files, num_samples))
post_joint_eigs2 = np.zeros((num_files, num_samples))
post_joint_eigs3 = np.zeros((num_files, num_samples))

eig_paths = []
for i in range(num_files):
    eig_path = f"{data_path}/eigenvalues/{i}.csv"
    if os.path.exists(eig_path):
        eig_paths.append(eig_path)

    distance_data = pd.read_csv(eig_path)
    pre_eigs1[i,:] = distance_data['pre_eig1']
    pre_eigs2[i,:] = distance_data['pre_eig2']
    pre_eigs3[i,:] = distance_data['pre_eig3']

    post_eigs1[i,:] = distance_data['post_eig1']
    post_eigs2[i,:] = distance_data['post_eig2']
    post_eigs3[i,:] = distance_data['post_eig3']

    pre_joint_eigs1[i,:] = distance_data['pre_joint_eig1']
    pre_joint_eigs2[i,:] = distance_data['pre_joint_eig2']
    pre_joint_eigs3[i,:] = distance_data['pre_joint_eig3']

    post_joint_eigs1[i,:] = distance_data['post_joint_eig1']
    post_joint_eigs2[i,:] = distance_data['post_joint_eig2']
    post_joint_eigs3[i,:] = distance_data['post_joint_eig3']

# ------------------------------------------------------------------------------------#

# take the average and standard deviation of each of the eigenvalues
pre_eigs1_avg = np.mean(pre_eigs1, axis=1)
pre_eigs1_std = np.std(pre_eigs1, axis=1)

pre_eigs2_avg = np.mean(pre_eigs2, axis=1)
pre_eigs2_std = np.std(pre_eigs2, axis=1)

pre_eigs3_avg = np.mean(pre_eigs3, axis=1)
pre_eigs3_std = np.std(pre_eigs3, axis=1)

post_eigs1_avg = np.mean(post_eigs1, axis=1)
post_eigs1_std = np.std(post_eigs1, axis=1)

post_eigs2_avg = np.mean(post_eigs2, axis=1)
post_eigs2_std = np.std(post_eigs2, axis=1)

post_eigs3_avg = np.mean(post_eigs3, axis=1)
post_eigs3_std = np.std(post_eigs3, axis=1)

pre_joint_eigs1_avg = np.mean(pre_joint_eigs1, axis=1)
pre_joint_eigs1_std = np.std(pre_joint_eigs1, axis=1)

pre_joint_eigs2_avg = np.mean(pre_joint_eigs2, axis=1)
pre_joint_eigs2_std = np.std(pre_joint_eigs2, axis=1)

pre_joint_eigs3_avg = np.mean(pre_joint_eigs3, axis=1)
pre_joint_eigs3_std = np.std(pre_joint_eigs3, axis=1)

post_joint_eigs1_avg = np.mean(post_joint_eigs1, axis=1)
post_joint_eigs1_std = np.std(post_joint_eigs1, axis=1)

post_joint_eigs2_avg = np.mean(post_joint_eigs2, axis=1)
post_joint_eigs2_std = np.std(post_joint_eigs2, axis=1)

post_joint_eigs3_avg = np.mean(post_joint_eigs3, axis=1)
post_joint_eigs3_std = np.std(post_joint_eigs3, axis=1)

# ------------------------------------------------------------------------------------#
# now plot the eigenvalues of the pre and post ablation populations in 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Color map for E/I ratio
normalize = mcolors.Normalize(vmin=np.min(e_to_i_range), vmax=np.max(e_to_i_range))
colors_pre = plt.cm.summer(normalize(e_to_i_range))
colors_post = plt.cm.hot(normalize(e_to_i_range))

# Scatter plot for eigenvalues
sc = ax.scatter(pre_eigs1_avg, pre_eigs2_avg, pre_eigs3_avg, c=colors_pre, label='Pre Ablation', marker='o')
sc = ax.scatter(post_eigs1_avg, post_eigs2_avg, post_eigs3_avg, c=colors_post, label='Post Ablation', marker='o')

# Simulating error bars with translucent spheres around each point
size_multiplier = 200  # Reduced size factor
for i in range(len(pre_eigs1_avg)):
    ax.scatter([pre_eigs1_avg[i]], [pre_eigs2_avg[i]], [pre_eigs3_avg[i]], 
               s=size_multiplier * pre_eigs1_std[i], alpha=0.1, color=colors_pre[i], marker='o')
    ax.scatter([pre_eigs1_avg[i]], [pre_eigs2_avg[i]], [pre_eigs3_avg[i]], 
               s=size_multiplier * pre_eigs2_std[i], alpha=0.1, color=colors_pre[i], marker='o')
    ax.scatter([pre_eigs1_avg[i]], [pre_eigs2_avg[i]], [pre_eigs3_avg[i]], 
               s=size_multiplier * pre_eigs3_std[i], alpha=0.1, color=colors_pre[i], marker='o')
    
    ax.scatter([post_eigs1_avg[i]], [post_eigs2_avg[i]], [post_eigs3_avg[i]],
                s=size_multiplier * post_eigs1_std[i], alpha=0.1, color=colors_post[i], marker='o')
    ax.scatter([post_eigs1_avg[i]], [post_eigs2_avg[i]], [post_eigs3_avg[i]],
                s=size_multiplier * post_eigs2_std[i], alpha=0.1, color=colors_post[i], marker='o')
    ax.scatter([post_eigs1_avg[i]], [post_eigs2_avg[i]], [post_eigs3_avg[i]],
                s=size_multiplier * post_eigs3_std[i], alpha=0.1, color=colors_post[i], marker='o')
    

# Set labels
ax.set_xlabel('1st λ Average')
ax.set_ylabel('2nd λ Average')
ax.set_zlabel('3rd λ Average')

# Setting axis limits for the first octant
ax.set_xlim(0, max(pre_eigs1_avg.max(), pre_eigs1_std.max()))
ax.set_ylim(0, max(pre_eigs2_avg.max(), pre_eigs2_std.max()))
ax.set_zlim(0, max(pre_eigs3_avg.max(), pre_eigs3_std.max()))

ax.legend()

# Color bar
sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=normalize)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.05, shrink=0.5)
cbar.set_label('E/I Ratio, Post Ablation')

sm = plt.cm.ScalarMappable(cmap=plt.cm.summer, norm=normalize)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.5)
cbar.set_label('E/I Ratio, Pre Ablation')

plt.title('Top Three Eigenvalues of Excitatory Population Activity')
plt.savefig(path + "/exc_eigenvalues.png", dpi = 300)

# ------------------------------------------------------------------------------------#
# now do this only for the exc population
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Color map for E/I ratio
normalize = mcolors.Normalize(vmin=np.min(e_to_i_range), vmax=np.max(e_to_i_range))
colors_pre = plt.cm.summer(normalize(e_to_i_range))

# Scatter plot for eigenvalues
sc = ax.scatter(pre_eigs1_avg, pre_eigs2_avg, pre_eigs3_avg, c=colors_pre, label='Pre Ablation', marker='o')

# Simulating error bars with translucent spheres around each point
size_multiplier = 200  # Reduced size factor
for i in range(len(pre_eigs1_avg)):
    ax.scatter([pre_eigs1_avg[i]], [pre_eigs2_avg[i]], [pre_eigs3_avg[i]], 
               s=size_multiplier * pre_eigs1_std[i], alpha=0.1, color=colors_pre[i], marker='o')
    ax.scatter([pre_eigs1_avg[i]], [pre_eigs2_avg[i]], [pre_eigs3_avg[i]], 
               s=size_multiplier * pre_eigs2_std[i], alpha=0.1, color=colors_pre[i], marker='o')
    ax.scatter([pre_eigs1_avg[i]], [pre_eigs2_avg[i]], [pre_eigs3_avg[i]], 
               s=size_multiplier * pre_eigs3_std[i], alpha=0.1, color=colors_pre[i], marker='o')

# Set labels
ax.set_xlabel('1st λ Average')
ax.set_ylabel('2nd λ Average')
ax.set_zlabel('3rd λ Average')

# Setting axis limits for the first octant
ax.set_xlim(0, max(pre_eigs1_avg.max(), pre_eigs1_std.max()))
ax.set_ylim(0, max(pre_eigs2_avg.max(), pre_eigs2_std.max()))
ax.set_zlim(0, max(pre_eigs3_avg.max(), pre_eigs3_std.max()))

ax.legend()

# Color bar
sm = plt.cm.ScalarMappable(cmap=plt.cm.summer, norm=normalize)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.05, shrink=0.5)
cbar.set_label('E/I Ratio, Pre Ablation')

plt.title('Top Three Eigenvalues of Excitatory Population Activity')
plt.savefig(path + "/exc_eigenvalues_pre.png", dpi = 300)

# ------------------------------------------------------------------------------------#
# now do this only for the ablated population
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Color map for E/I ratio
normalize = mcolors.Normalize(vmin=np.min(e_to_i_range), vmax=np.max(e_to_i_range))
colors_post = plt.cm.hot(normalize(e_to_i_range))

# Scatter plot for eigenvalues
sc = ax.scatter(post_eigs1_avg, post_eigs2_avg, post_eigs3_avg, c=colors_post, label='Post Ablation', marker='o')

# Simulating error bars with translucent spheres around each point
size_multiplier = 200  # Reduced size factor
for i in range(len(post_eigs1_avg)):
    ax.scatter([post_eigs1_avg[i]], [post_eigs2_avg[i]], [post_eigs3_avg[i]],
                s=size_multiplier * post_eigs1_std[i], alpha=0.1, color=colors_post[i], marker='o')
    ax.scatter([post_eigs1_avg[i]], [post_eigs2_avg[i]], [post_eigs3_avg[i]],
                s=size_multiplier * post_eigs2_std[i], alpha=0.1, color=colors_post[i], marker='o')
    ax.scatter([post_eigs1_avg[i]], [post_eigs2_avg[i]], [post_eigs3_avg[i]],
                s=size_multiplier * post_eigs3_std[i], alpha=0.1, color=colors_post[i], marker='o')

# Set labels
ax.set_xlabel('1st λ Average')
ax.set_ylabel('2nd λ Average')
ax.set_zlabel('3rd λ Average')

# Setting axis limits for the first octant
ax.set_xlim(0, max(post_eigs1_avg.max(), post_eigs1_std.max()))
ax.set_ylim(0, max(post_eigs2_avg.max(), post_eigs2_std.max()))
ax.set_zlim(0, max(post_eigs3_avg.max(), post_eigs3_std.max()))

ax.legend()

# Color bar
sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=normalize)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.05, shrink=0.5)
cbar.set_label('E/I Ratio, Post Ablation')

plt.title('Top Three Eigenvalues of Excitatory Population Activity')
plt.savefig(path + "/exc_eigenvalues_post.png", dpi = 300)

# ----------------------------------------------------------------------------------------------- #
# do the same thing for the joint population
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Color map for E/I ratio
normalize = mcolors.Normalize(vmin=np.min(e_to_i_range), vmax=np.max(e_to_i_range))
colors_pre = plt.cm.summer(normalize(e_to_i_range))
colors_post = plt.cm.hot(normalize(e_to_i_range))

# Scatter plot for eigenvalues
sc = ax.scatter(pre_joint_eigs1_avg, pre_joint_eigs2_avg, pre_joint_eigs3_avg, c=colors_pre, label='Pre Ablation', marker='o')
sc = ax.scatter(post_joint_eigs1_avg, post_joint_eigs2_avg, post_joint_eigs3_avg, c=colors_post, label='Post Ablation', marker='o')

# Simulating error bars with translucent spheres around each point
size_multiplier = 200  # Reduced size factor
for i in range(len(pre_joint_eigs1_avg)):
    ax.scatter([pre_joint_eigs1_avg[i]], [pre_joint_eigs2_avg[i]], [pre_joint_eigs3_avg[i]], 
               s=size_multiplier * pre_joint_eigs1_std[i], alpha=0.1, color=colors_pre[i], marker='o')
    ax.scatter([pre_joint_eigs1_avg[i]], [pre_joint_eigs2_avg[i]], [pre_joint_eigs3_avg[i]], 
               s=size_multiplier * pre_joint_eigs2_std[i], alpha=0.1, color=colors_pre[i], marker='o')
    ax.scatter([pre_joint_eigs1_avg[i]], [pre_joint_eigs2_avg[i]], [pre_joint_eigs3_avg[i]], 
               s=size_multiplier * pre_joint_eigs3_std[i], alpha=0.1, color=colors_pre[i], marker='o')
    
    ax.scatter([post_joint_eigs1_avg[i]], [post_joint_eigs2_avg[i]], [post_joint_eigs3_avg[i]],
                s=size_multiplier * post_joint_eigs1_std[i], alpha=0.1, color=colors_post[i], marker='o')
    ax.scatter([post_joint_eigs1_avg[i]], [post_joint_eigs2_avg[i]], [post_joint_eigs3_avg[i]],
                s=size_multiplier * post_joint_eigs2_std[i], alpha=0.1, color=colors_post[i], marker='o')
    ax.scatter([post_joint_eigs1_avg[i]], [post_joint_eigs2_avg[i]], [post_joint_eigs3_avg[i]],
                s=size_multiplier * post_joint_eigs3_std[i], alpha=0.1, color=colors_post[i], marker='o')
# Set labels
ax.set_xlabel('1st λ Average')
ax.set_ylabel('2nd λ Average')
ax.set_zlabel('3rd λ Average')

# Setting axis limits for the first octant
ax.set_xlim(0, max(pre_joint_eigs1_avg.max(), pre_joint_eigs1_std.max()))
ax.set_ylim(0, max(pre_joint_eigs2_avg.max(), pre_joint_eigs2_std.max()))
ax.set_zlim(0, max(pre_joint_eigs3_avg.max(), pre_joint_eigs3_std.max()))

ax.legend()

# Color bar
sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=normalize)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.05, shrink=0.5)
cbar.set_label('E/I Ratio, Post Ablation')

sm = plt.cm.ScalarMappable(cmap=plt.cm.summer, norm=normalize)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.5)
cbar.set_label('E/I Ratio, Pre Ablation')

plt.title('Top Three Eigenvalues of Joint Population Activity')
plt.savefig(path + "/joint_eigenvalues.png", dpi = 300)

# ------------------------------------------------------------------------------------#
# now do this only for the exc population
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Color map for E/I ratio
normalize = mcolors.Normalize(vmin=np.min(e_to_i_range), vmax=np.max(e_to_i_range))
colors_pre = plt.cm.summer(normalize(e_to_i_range))

# Scatter plot for eigenvalues
sc = ax.scatter(pre_joint_eigs1_avg, pre_joint_eigs2_avg, pre_joint_eigs3_avg, c=colors_pre, label='Pre Ablation', marker='o')

# Simulating error bars with translucent spheres around each point
size_multiplier = 200  # Reduced size factor
for i in range(len(pre_joint_eigs1_avg)):
    ax.scatter([pre_joint_eigs1_avg[i]], [pre_joint_eigs2_avg[i]], [pre_joint_eigs3_avg[i]],
                s=size_multiplier * pre_joint_eigs1_std[i], alpha=0.1, color=colors_pre[i], marker='o')
    ax.scatter([pre_joint_eigs1_avg[i]], [pre_joint_eigs2_avg[i]], [pre_joint_eigs3_avg[i]],
                s=size_multiplier * pre_joint_eigs2_std[i], alpha=0.1, color=colors_pre[i], marker='o')
    ax.scatter([pre_joint_eigs1_avg[i]], [pre_joint_eigs2_avg[i]], [pre_joint_eigs3_avg[i]],
                s=size_multiplier * pre_joint_eigs3_std[i], alpha=0.1, color=colors_pre[i], marker='o')
    
# Set labels
ax.set_xlabel('1st λ Average')
ax.set_ylabel('2nd λ Average')
ax.set_zlabel('3rd λ Average')

# Setting axis limits for the first octant
ax.set_xlim(0, max(pre_joint_eigs1_avg.max(), pre_joint_eigs1_std.max()))
ax.set_ylim(0, max(pre_joint_eigs2_avg.max(), pre_joint_eigs2_std.max()))
ax.set_zlim(0, max(pre_joint_eigs3_avg.max(), pre_joint_eigs3_std.max()))

ax.legend()

# Color bar
sm = plt.cm.ScalarMappable(cmap=plt.cm.summer, norm=normalize)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.05, shrink=0.5)
cbar.set_label('E/I Ratio, Pre Ablation')

plt.title('Top Three Eigenvalues of Joint Population Activity')
plt.savefig(path + "/joint_eigenvalues_pre.png", dpi = 300)

# ------------------------------------------------------------------------------------#
# now do this only for the ablated population
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Color map for E/I ratio
normalize = mcolors.Normalize(vmin=np.min(e_to_i_range), vmax=np.max(e_to_i_range))
colors_post = plt.cm.hot(normalize(e_to_i_range))

# Scatter plot for eigenvalues
sc = ax.scatter(post_joint_eigs1_avg, post_joint_eigs2_avg, post_joint_eigs3_avg, c=colors_post, label='Post Ablation', marker='o')

# Simulating error bars with translucent spheres around each point
size_multiplier = 200  # Reduced size factor
for i in range(len(post_joint_eigs1_avg)):
    ax.scatter([post_joint_eigs1_avg[i]], [post_joint_eigs2_avg[i]], [post_joint_eigs3_avg[i]],
                s=size_multiplier * post_joint_eigs1_std[i], alpha=0.1, color=colors_post[i], marker='o')
    ax.scatter([post_joint_eigs1_avg[i]], [post_joint_eigs2_avg[i]], [post_joint_eigs3_avg[i]],
                s=size_multiplier * post_joint_eigs2_std[i], alpha=0.1, color=colors_post[i], marker='o')
    ax.scatter([post_joint_eigs1_avg[i]], [post_joint_eigs2_avg[i]], [post_joint_eigs3_avg[i]],
                s=size_multiplier * post_joint_eigs3_std[i], alpha=0.1, color=colors_post[i], marker='o')
    
# Set labels
ax.set_xlabel('1st λ Average')
ax.set_ylabel('2nd λ Average')
ax.set_zlabel('3rd λ Average')

# Setting axis limits for the first octant
ax.set_xlim(0, max(post_joint_eigs1_avg.max(), post_joint_eigs1_std.max()))
ax.set_ylim(0, max(post_joint_eigs2_avg.max(), post_joint_eigs2_std.max()))
ax.set_zlim(0, max(post_joint_eigs3_avg.max(), post_joint_eigs3_std.max()))

ax.legend()

# Color bar
sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=normalize)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.05, shrink=0.5)
cbar.set_label('E/I Ratio, Post Ablation')

plt.title('Top Three Eigenvalues of Joint Population Activity')
plt.savefig(path + "/joint_eigenvalues_post.png", dpi = 300)

# ----------------------------------------------------------------------------------------------- #
##### NOTE: display/close functionality for plotting ease #####
plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run
# ----------------------------------------------------------------------------------------------- # 
