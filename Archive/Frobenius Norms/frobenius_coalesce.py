import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

os.environ
OUT_DIR = "/Users/Vassilis/Desktop/BIL/Winter 24 Writeup/Images from Data" #NOTE: change this for output folder
os.system('mkdir {}'.format(os.path.join(OUT_DIR,'plots')))

# azimuth = -95, elevation= 41

name = "Continuous Tracking" # NOTE: change this for the name of the experiment
encoders = False # NOTE: change this for the type of experiment
accuracy = False

data_path = f"/Users/Vassilis/Desktop/BIL/Winter 24 Writeup/Data/Continuous_tracking"
path = os.path.join(OUT_DIR,f'{name}')
if not os.path.exists(path):
        os.makedirs(path)

e_to_i_range = np.arange(0.1, 0.91, 0.01) # 81 different parameter values
num_files = len(e_to_i_range)

frobenius_range = 200 # number of points to sample in the frobenius distance range

num_iter = 21 #30
num_samples = int((num_iter *(num_iter - 1))/2)

num_sources = 4

pre_data = np.zeros((num_files, num_samples))
post_data = np.zeros((num_files, num_samples))
diff_data = np.zeros((num_files, num_samples))

pre_joint_data = np.zeros((num_files, num_samples))
post_joint_data = np.zeros((num_files, num_samples))
diff_joint_data = np.zeros((num_files, num_samples))

if encoders:
    pre_exc_encoders_data = np.zeros((num_files, num_samples))
    pre_inh_encoders_data = np.zeros((num_files, num_samples))
    post_exc_encoders_data = np.zeros((num_files, num_samples))
    post_inh_encoders_data = np.zeros((num_files, num_samples))

pre_accuracy = np.zeros((num_files, num_iter))
post_accuracy = np.zeros((num_files, num_iter))
diff_accuracy = np.zeros((num_files, num_iter))

joint_pre_accuracy = np.zeros((num_files, num_iter))
joint_post_accuracy = np.zeros((num_files, num_iter))
joint_diff_accuracy = np.zeros((num_files, num_iter))

distance_paths = []
accuracy_paths = []
for i in range(num_files):
    distance_path = f"{data_path}/distances/{i}.csv"
    if os.path.exists(distance_path):
        distance_paths.append(distance_path)

    distance_data = pd.read_csv(distance_path)
    pre_data[i,:] = distance_data['pre']
    post_data[i,:] = distance_data['post']
    diff_data[i,:] = distance_data['diff']

    pre_joint_data[i,:] = distance_data['pre_joint']
    post_joint_data[i,:] = distance_data['post_joint']
    diff_joint_data[i,:] = distance_data['diff_joint']

    if encoders:
        pre_exc_encoders_data[i,:] = distance_data['pre_exc_enc']
        pre_inh_encoders_data[i,:] = distance_data['pre_inh_enc']
        post_exc_encoders_data[i,:] = distance_data['post_exc_enc']
        post_inh_encoders_data[i,:] = distance_data['post_inh_enc']

    if accuracy:
        accuracy_path = f"{data_path}/accuracy/{i}.csv"
        if os.path.exists(accuracy_path):
            accuracy_paths.append(accuracy_path)
        
        accuracy_data = pd.read_csv(accuracy_path)
        pre_accuracy[i,:] = accuracy_data['pre']
        post_accuracy[i,:] = accuracy_data['post']
        diff_accuracy[i,:] = accuracy_data['diff']

        joint_pre_accuracy[i,:] = accuracy_data['joint_pre']
        joint_post_accuracy[i,:] = accuracy_data['joint_post']
        joint_diff_accuracy[i,:] = accuracy_data['joint_diff']

# ------------------------------------------------------------------------------------#

def compute_density(x_axis, data):
    d_data = np.zeros((num_files, frobenius_range))

    data += 0.01

    for i in range(data.shape[0]):
        density = gaussian_kde(data[i,:])
        density.covariance_factor = lambda: .25
        density._compute_covariance()
        d_data[i,:] = density(x_axis)

    return d_data

def plot_3d_density(x_axis, y_axis, d_data, d_accuracy, title, color):
    x_repeated = np.tile(x_axis, len(y_axis))
    y_repeated = np.repeat(y_axis, len(x_axis))
    z_flattened = d_data.flatten('C')  # Flatten the 2D array in column-major order

    # Creating a new DataFrame with the structured data
    df_long = pd.DataFrame({'x': x_repeated, 'y': y_repeated, 'z': z_flattened})

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Frobenius Distance')
    ax.set_ylabel('E/I Ratio')
    ax.set_zlabel('Density')

    # include the accuracy of the model as a scatter plot
    y = y_axis

    if d_accuracy is not None:
        z = np.mean(d_accuracy, axis=1)
        z = np.array(z) * 0.4 #d_data.max() # scale the accuracy to the density so that it is visible
        ax.scatter(y, z, zs=0, zdir='x', label='Model Performance', color=color)

    # Using the restructured DataFrame for plotting
    ax.scatter3D(df_long.x, df_long.y, df_long.z, zdir='z',cmap='viridis', norm=norm)
    ax.set_zlim(0, 0.4)
    surf = ax.plot_trisurf(df_long.x, df_long.y, df_long.z, cmap=cm.jet, linewidth=0.1, norm=norm)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.legend(loc='upper right')
    plt.title(f'{title}, FDD')
    plt.savefig(path + f"/{title}, FDD.png", dpi=300)
    ax.view_init(elev=41, azim=-95) # rotate the plot
    plt.savefig(path + f"/{title}, FDD_rotated.png", dpi=300)

# ------------------------------------------------------------------------------------#
# NOTE: The following code is for plotting the densities in 3D for the pre, post, and diff data
mins = [pre_data.min(), post_data.min(), diff_data.min()]
maxs = [pre_data.max(), post_data.max(), diff_data.max()]

y_axis = e_to_i_range
x_axis = np.linspace(min(mins), max(maxs), frobenius_range)

d_pre = compute_density(x_axis, pre_data)
d_post = compute_density(x_axis, post_data)
d_diff = compute_density(x_axis, diff_data)

max_density = max(d_pre.max(), d_post.max(), d_diff.max())
norm = Normalize(vmin=0, vmax=0.4)

if accuracy:
    plot_3d_density(x_axis, y_axis, d_pre, pre_accuracy, "pre, exc pop", "green")
    plot_3d_density(x_axis, y_axis, d_post, post_accuracy, "post, exc pop", "red")
    plot_3d_density(x_axis, y_axis, d_diff, diff_accuracy, "pre - post, exc pop", "purple")
else:
    plot_3d_density(x_axis, y_axis, d_pre, None, "pre, exc pop", "green")
    plot_3d_density(x_axis, y_axis, d_post, None, "post, exc pop", "red")
    plot_3d_density(x_axis, y_axis, d_diff, None, "pre - post, exc pop", "purple")

# ------------------------------------------------------------------------------------#
# NOTE: The following code is for plotting the densities in 3D for the joint data
mins_joint = [pre_joint_data.min(), post_joint_data.min(), diff_joint_data.min()]
maxs_joint = [pre_joint_data.max(), post_joint_data.max(), diff_joint_data.max()]

x_axis_joint = np.linspace(min(mins_joint), max(maxs_joint), frobenius_range)

d_pre_joint = compute_density(x_axis_joint, pre_joint_data)
d_post_joint = compute_density(x_axis_joint, post_joint_data)
d_diff_joint = compute_density(x_axis_joint, diff_joint_data)

max_density_joint = max(d_pre_joint.max(), d_post_joint.max(), d_diff_joint.max())
norm_joint = Normalize(vmin=0, vmax=max_density_joint)

if accuracy:
    plot_3d_density(x_axis_joint, y_axis, d_pre_joint, joint_pre_accuracy, "pre, entire pop", "green")
    plot_3d_density(x_axis_joint, y_axis, d_post_joint, joint_post_accuracy,  "post, entire pop", "red")
    plot_3d_density(x_axis_joint, y_axis, d_diff_joint, joint_diff_accuracy, "pre - post, entire pop", "purple")
else:
    plot_3d_density(x_axis_joint, y_axis, d_pre_joint, None, "pre, entire pop", "green")
    plot_3d_density(x_axis_joint, y_axis, d_post_joint, None, "post, entire pop", "red")
    plot_3d_density(x_axis_joint, y_axis, d_diff_joint, None, "pre - post, entire pop", "purple")
# ------------------------------------------------------------------------------------#
#NOTE: The following code is for plotting the encoder densities of the model

if encoders:
    mins_enc = [pre_exc_encoders_data.min(), pre_inh_encoders_data.min(), post_exc_encoders_data.min(), post_inh_encoders_data.min()]
    maxs_enc = [pre_exc_encoders_data.max(), pre_inh_encoders_data.max(), post_exc_encoders_data.max(), post_inh_encoders_data.max()]

    x_axis_enc = np.linspace(min(mins_enc), max(maxs_enc), frobenius_range)

    d_pre_exc_enc = compute_density(x_axis_enc, pre_exc_encoders_data)
    d_pre_inh_enc = compute_density(x_axis_enc, pre_inh_encoders_data)
    d_post_exc_enc = compute_density(x_axis_enc, post_exc_encoders_data)
    d_post_inh_enc = compute_density(x_axis_enc, post_inh_encoders_data)

    plot_3d_density(x_axis_enc, y_axis, d_pre_exc_enc, None, "pre, exc enc", "green")
    plot_3d_density(x_axis_enc, y_axis, d_pre_inh_enc, None, "pre, inh enc", "green")
    plot_3d_density(x_axis_enc, y_axis, d_post_exc_enc, None, "post, exc enc", "red")
    plot_3d_density(x_axis_enc, y_axis, d_post_inh_enc, None, "post, inh enc", "red")
# ------------------------------------------------------------------------------------#

# fig, axs = plt.subplots(1, 3, figsize=(18,6))
# axs = axs.flatten()
# for ax in axs: ax.axis("off") # remove the markings from the original figure
# axs[0].imshow(plt.imread(path + "/pre, exc pop, FDD.png"), aspect='auto')
# axs[1].imshow(plt.imread(path + "/post, exc pop, FDD.png"), aspect='auto')
# axs[2].imshow(plt.imread(path + "/pre - post, exc pop, FDD.png"), aspect='auto')

# fig.suptitle(f"Excitatory Population FDD, {num_sources} Sources", fontsize = 10)
# fig.savefig(path + "/FDD_exc_pop.png", dpi = 300)

# plt.tight_layout()

# fig, axs = plt.subplots(1, 3, figsize=(18,6))
# axs = axs.flatten()
# for ax in axs: ax.axis("off") # remove the markings from the original figure
# axs[0].imshow(plt.imread(path + "/pre, entire pop, FDD.png"), aspect='auto')
# axs[1].imshow(plt.imread(path + "/post, entire pop, FDD.png"), aspect='auto')
# axs[2].imshow(plt.imread(path + "/pre - post, entire pop, FDD.png"), aspect='auto')

# fig.suptitle(f"Entire Population FDD, {num_sources} Sources", fontsize = 10)
# fig.savefig(path + "/FDD_entire_pop.png", dpi = 300)

# plt.tight_layout()

# fig, axs = plt.subplots(1, 3, figsize=(18,6))
# axs = axs.flatten()
# for ax in axs: ax.axis("off") # remove the markings from the original figure
# axs[0].imshow(plt.imread(path + "/pre, exc pop, FDD_rotated.png"), aspect='auto')
# axs[1].imshow(plt.imread(path + "/post, exc pop, FDD_rotated.png"), aspect='auto')
# axs[2].imshow(plt.imread(path + "/pre - post, exc pop, FDD_rotated.png"), aspect='auto')

# fig.suptitle(f"Excitatory Population FDD, {num_sources} Sources", fontsize = 10)
# fig.savefig(path + "/FDD_exc_pop_rotated.png", dpi = 300)

# plt.tight_layout()

# fig, axs = plt.subplots(1, 3, figsize=(18,6))
# axs = axs.flatten()
# for ax in axs: ax.axis("off") # remove the markings from the original figure
# axs[0].imshow(plt.imread(path + "/pre, entire pop, FDD_rotated.png"), aspect='auto')
# axs[1].imshow(plt.imread(path + "/post, entire pop, FDD_rotated.png"), aspect='auto')
# axs[2].imshow(plt.imread(path + "/pre - post, entire pop, FDD_rotated.png"), aspect='auto')

# fig.suptitle(f"Entire Population FDD, {num_sources} Sources", fontsize = 10)
# fig.savefig(path + "/FDD_entire_pop_rotated.png", dpi = 300)

# ------------------------------------------------------------------------------------#

# now make a 2D plot of the accuracy
fig, ax = plt.subplots()
pre_accuracy_mean = np.mean(pre_accuracy, axis=1)
pre_accuracy_std = np.std(pre_accuracy, axis=1)
post_accuracy_mean = np.mean(post_accuracy, axis=1)
post_accuracy_std = np.std(post_accuracy, axis=1)
diff_accuracy_mean = np.mean(diff_accuracy, axis=1)
diff_accuracy_std = np.std(diff_accuracy, axis=1)

ax.plot(y_axis, pre_accuracy_mean, label='pre ablation', color='green')
ax.fill_between(y_axis, pre_accuracy_mean - pre_accuracy_std, pre_accuracy_mean + pre_accuracy_std, color='green', alpha=0.2)
ax.plot(y_axis, post_accuracy_mean, label='post ablation', color='red')
ax.fill_between(y_axis, post_accuracy_mean - post_accuracy_std, post_accuracy_mean + post_accuracy_std, color='red', alpha=0.2)
# ax.plot(y_axis, diff_accuracy_mean, label='difference', color='purple')
# ax.fill_between(y_axis, diff_accuracy_mean - diff_accuracy_std, diff_accuracy_mean + diff_accuracy_std, color='purple', alpha=0.2)

ax.set_xlabel('E/I Ratio')
ax.set_ylabel('Average Classification Accuracy (across trials and sources)')
ax.set_title(f'{name} Excitatory Population')
plt.legend(loc='lower left')
plt.savefig(path + f"/classification_accuracy.png", dpi=300)

# ------------------------------------------------------------------------------------#
fig, ax = plt.subplots()
joint_pre_accuracy_mean = np.mean(joint_pre_accuracy, axis=1)
joint_pre_accuracy_std = np.std(joint_pre_accuracy, axis=1)
joint_post_accuracy_mean = np.mean(joint_post_accuracy, axis=1)
joint_post_accuracy_std = np.std(joint_post_accuracy, axis=1)
joint_diff_accuracy_mean = np.mean(joint_diff_accuracy, axis=1)
joint_diff_accuracy_std = np.std(joint_diff_accuracy, axis=1)

ax.plot(y_axis, joint_pre_accuracy_mean, label='pre ablation', color='olive', linestyle='dashed')
ax.fill_between(y_axis, joint_pre_accuracy_mean - joint_pre_accuracy_std, joint_pre_accuracy_mean + joint_pre_accuracy_std, color='olive', alpha=0.2)
ax.plot(y_axis, joint_post_accuracy_mean, label='post ablation', color='darkorange', linestyle='dashed')
ax.fill_between(y_axis, joint_post_accuracy_mean - joint_post_accuracy_std, joint_post_accuracy_mean + joint_post_accuracy_std, color='darkorange', alpha=0.2)
# ax.plot(y_axis, joint_diff_accuracy_mean, label='difference', color='purple')
# ax.fill_between(y_axis, joint_diff_accuracy_mean - joint_diff_accuracy_std, joint_diff_accuracy_mean + joint_diff_accuracy_std, color='purple', alpha=0.2)

ax.set_xlabel('E/I Ratio')
ax.set_ylabel('Average Classification Accuracy (across trials and sources)')
ax.set_title(f'{name} Entire Population')
plt.legend(loc='lower left')
plt.savefig(path + f"/classification_accuracy_joint.png", dpi=300)

# now plot them together
fig, ax = plt.subplots()
ax.plot(y_axis, pre_accuracy_mean, label='pre ablation', color='green')
ax.fill_between(y_axis, pre_accuracy_mean - pre_accuracy_std, pre_accuracy_mean + pre_accuracy_std, color='green', alpha=0.2)
ax.plot(y_axis, joint_pre_accuracy_mean, label='pre ablation', color='olive', linestyle='dashed')
ax.fill_between(y_axis, joint_pre_accuracy_mean - joint_pre_accuracy_std, joint_pre_accuracy_mean + joint_pre_accuracy_std, color='olive', alpha=0.2)

ax.plot(y_axis, post_accuracy_mean, label='post ablation', color='red')
ax.fill_between(y_axis, post_accuracy_mean - post_accuracy_std, post_accuracy_mean + post_accuracy_std, color='red', alpha=0.2)
ax.plot(y_axis, joint_post_accuracy_mean, label='post ablation', color='darkorange', linestyle='dashed')
ax.fill_between(y_axis, joint_post_accuracy_mean - joint_post_accuracy_std, joint_post_accuracy_mean + joint_post_accuracy_std, color='darkorange', alpha=0.2)

ax.set_xlabel('E/I Ratio')
ax.set_ylabel('Average Classification Accuracy (across trials and sources)')
ax.set_title(f'{name} Excitatory + Entire Population')
plt.legend(loc='lower left')
plt.savefig(path + f"/classification_accuracy_together.png", dpi=300)

# ------------------------------------------------------------------------------------#
# Now make a 2D plot of the distance standard deviation
pre_mean = np.mean(pre_data, axis=1)
pre_std = np.std(pre_data, axis=1)
post_mean = np.mean(post_data, axis=1)
post_std = np.std(post_data, axis=1)
diff_mean = np.mean(diff_data, axis=1)
diff_std = np.std(diff_data, axis=1)

pre_joint_mean = np.mean(pre_joint_data, axis=1)
pre_joint_std = np.std(pre_joint_data, axis=1)
post_joint_mean = np.mean(post_joint_data, axis=1)
post_joint_std = np.std(post_joint_data, axis=1)
diff_joint_mean = np.mean(diff_joint_data, axis=1)
diff_joint_std = np.std(diff_joint_data, axis=1)

if encoders:
    pre_exc_encoders_mean = np.mean(pre_exc_encoders_data, axis=1)
    pre_exc_encoders_std = np.std(pre_exc_encoders_data, axis=1)
    pre_inh_encoders_mean = np.mean(pre_inh_encoders_data, axis=1)
    pre_inh_encoders_std = np.std(pre_inh_encoders_data, axis=1)

    post_exc_encoders_mean = np.mean(post_exc_encoders_data, axis=1)
    post_exc_encoders_std = np.std(post_exc_encoders_data, axis=1)
    post_inh_encoders_mean = np.mean(post_inh_encoders_data, axis=1)
    post_inh_encoders_std = np.std(post_inh_encoders_data, axis=1)

fig, ax = plt.subplots()
ax.plot(y_axis, pre_mean, label='pre', color='green')
ax.fill_between(y_axis, pre_mean - pre_std, pre_mean + pre_std, color='green', alpha=0.2)
ax.plot(y_axis, pre_joint_mean, label='pre joint', color='olive', linestyle='dashed')
ax.fill_between(y_axis, pre_joint_mean - pre_joint_std, pre_joint_mean + pre_joint_std, color='olive', alpha=0.2)

ax.plot(y_axis, post_mean, label='post', color='red')
ax.fill_between(y_axis, post_mean - post_std, post_mean + post_std, color='red', alpha=0.2)
ax.plot(y_axis, post_joint_mean, label='post joint', color='darkorange', linestyle='dashed')
ax.fill_between(y_axis, post_joint_mean - post_joint_std, post_joint_mean + post_joint_std, color='darkorange', alpha=0.2)

# ax.plot(y_axis, diff_mean, label='diff', color='purple')
# ax.fill_between(y_axis, diff_mean - diff_std, diff_mean + diff_std, color='purple', alpha=0.2)
# ax.plot(y_axis, diff_joint_mean, label='diff joint', color='darkmagenta', linestyle='dashed')
# ax.fill_between(y_axis, diff_joint_mean - diff_joint_std, diff_joint_mean + diff_joint_std, color='darkmagenta', alpha=0.2)

ax.set_xlabel('E/I Ratio')
ax.set_ylabel('Average Frobenius Distance')
ax.set_title(f'{name} Excitatory + Entire Population')
plt.legend(loc='upper left')
plt.savefig(path + f"/frobenius_averages_together.png", dpi=300)

fig, ax = plt.subplots()
ax.plot(y_axis, pre_mean, label='pre', color='green')
ax.fill_between(y_axis, pre_mean - pre_std, pre_mean + pre_std, color='green', alpha=0.2)
ax.plot(y_axis, post_mean, label='post', color='red')
ax.fill_between(y_axis, post_mean - post_std, post_mean + post_std, color='red', alpha=0.2)
# ax.plot(y_axis, diff_mean, label='diff', color='purple')
# ax.fill_between(y_axis, diff_mean - diff_std, diff_mean + diff_std, color='purple', alpha=0.2)
ax.set_xlabel('E/I Ratio')
ax.set_ylabel('Average Frobenius Distance')
ax.set_title(f'{name} Excitatory Population')
plt.legend(loc='upper left')
plt.savefig(path + f"/frobenius_averages_exc_pop.png", dpi=300)

fig, ax = plt.subplots()
ax.plot(y_axis, pre_joint_mean, label='pre', color='olive', linestyle='dashed')
ax.fill_between(y_axis, pre_joint_mean - pre_joint_std, pre_joint_mean + pre_joint_std, color='olive', alpha=0.2)
ax.plot(y_axis, post_joint_mean, label='post', color='darkorange', linestyle='dashed')
ax.fill_between(y_axis, post_joint_mean - post_joint_std, post_joint_mean + post_joint_std, color='darkorange', alpha=0.2)
# ax.plot(y_axis, diff_joint_mean, label='diff', color='darkmagenta', linestyle='dashed')
# ax.fill_between(y_axis, diff_joint_mean - diff_joint_std, diff_joint_mean + diff_joint_std, color='darkmagenta', alpha=0.2)
ax.set_xlabel('E/I Ratio')
ax.set_ylabel('Average Frobenius Distance')
ax.set_title(f'{name} Entire Population')
plt.legend(loc='upper left')
plt.savefig(path + f"/frobenius_averages_entire_pop.png", dpi=300)

# ------------------------------------------------------------------------------------#
# NOTE: this part of the code plots the average distances of the encoders
if encoders:
    fig, ax = plt.subplots()
    ax.plot(y_axis, pre_exc_encoders_mean, label='pre exc', color='green')
    ax.fill_between(y_axis, pre_exc_encoders_mean - pre_exc_encoders_std, pre_exc_encoders_mean + pre_exc_encoders_std, color='green', alpha=0.2)
    ax.plot(y_axis, pre_inh_encoders_mean, label='pre inh', color='red')
    ax.fill_between(y_axis, pre_inh_encoders_mean - pre_inh_encoders_std, pre_inh_encoders_mean + pre_inh_encoders_std, color='red', alpha=0.2)

    ax.plot(y_axis, post_exc_encoders_mean, label='post exc', color='olive', linestyle='dashed')
    ax.fill_between(y_axis, post_exc_encoders_mean - post_exc_encoders_std, post_exc_encoders_mean + post_exc_encoders_std, color='olive', alpha=0.2)
    ax.plot(y_axis, post_inh_encoders_mean, label='post inh', color='darkorange', linestyle='dashed')
    ax.fill_between(y_axis, post_inh_encoders_mean - post_inh_encoders_std, post_inh_encoders_mean + post_inh_encoders_std, color='darkorange', alpha=0.2)
    ax.set_xlabel('E/I Ratio')
    ax.set_ylabel('Average Frobenius Distance')
    ax.set_title(f'{name} Encoders')
    plt.legend(loc='upper left')
    plt.savefig(path + f"/encoder_averages.png", dpi=300)

# ------------------------------------------------------------------------------------#
# NOTE: this part of the code plots the standard deviation of the frobenius distance across populations
fig, ax = plt.subplots()
ax.plot(y_axis, pre_std, label='pre', color='green')
ax.plot(y_axis, post_std, label='post', color='red')
ax.set_xlabel('E/I Ratio')
ax.set_ylabel('Standard Deviation of Frobenius Distance')
ax.set_title(f'{name} Excitatory Population')
plt.legend(loc='upper left')
plt.savefig(path + f"/frobenius_std_exc_pop.png", dpi=300)

fig, ax = plt.subplots()
ax.plot(y_axis, pre_joint_std, label='pre', color='olive', linestyle='dashed')
ax.plot(y_axis, post_joint_std, label='post', color='darkorange', linestyle='dashed')
ax.set_xlabel('E/I Ratio')
ax.set_ylabel('Standard Deviation of Frobenius Distance')
ax.set_title(f'{name} Entire Population')
plt.legend(loc='upper left')
plt.savefig(path + f"/frobenius_std_entire_pop.png", dpi=300)

fig, ax = plt.subplots()
ax.plot(y_axis, pre_std, label='pre', color='green')
ax.plot(y_axis, post_std, label='post', color='red')
ax.plot(y_axis, pre_joint_std, label='pre joint', color='olive', linestyle='dashed')
ax.plot(y_axis, post_joint_std, label='post joint', color='darkorange', linestyle='dashed')
ax.set_xlabel('E/I Ratio')
ax.set_ylabel('Standard Deviation of Frobenius Distance')
ax.set_title(f'{name} Excitatory + Entire Population')
plt.legend(loc='upper left')
plt.savefig(path + f"/frobenius_std_together.png", dpi=300)

# ------------------------------------------------------------------------------------#
# NOTE: this part of the code makes the final panel figure including all the plots
fig, axs = plt.subplots(3, 4, figsize=(18,10))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure
axs[0].imshow(plt.imread(path + "/pre, exc pop, FDD_rotated.png"), aspect='auto')
axs[1].imshow(plt.imread(path + "/post, exc pop, FDD_rotated.png"), aspect='auto')
axs[2].imshow(plt.imread(path + "/frobenius_averages_exc_pop.png"), aspect='auto')
axs[3].imshow(plt.imread(path + "/classification_accuracy.png"), aspect='auto')

axs[4].imshow(plt.imread(path + "/pre, entire pop, FDD_rotated.png"), aspect='auto')
axs[5].imshow(plt.imread(path + "/post, entire pop, FDD_rotated.png"), aspect='auto')
axs[6].imshow(plt.imread(path + "/frobenius_averages_entire_pop.png"), aspect='auto')
axs[7].imshow(plt.imread(path + "/classification_accuracy_joint.png"), aspect='auto')

if encoders:
    axs[8].imshow(plt.imread(path + "/encoder_averages.png"), aspect='auto')
axs[9].imshow(plt.imread(path + "/frobenius_std_together.png"), aspect='auto')
axs[10].imshow(plt.imread(path + "/frobenius_averages_together.png"), aspect='auto')
axs[11].imshow(plt.imread(path + "/classification_accuracy_together.png"), aspect='auto')

fig.suptitle(f"{name}", fontsize = 10)
fig.tight_layout()  # Reduce whitespace and make the layout tighter
fig.savefig(path + "/panel.png", dpi = 300)

# ----------------------------------------------------------------------------------------------- #
##### NOTE: display/close functionality for plotting ease #####
plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run
# ----------------------------------------------------------------------------------------------- # 
