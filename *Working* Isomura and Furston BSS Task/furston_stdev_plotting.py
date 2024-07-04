import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ
OUT_DIR = "/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-02-02 Source Dynamics" #NOTE: change this for output folder
os.system('mkdir {}'.format(os.path.join(OUT_DIR,'plots')))

name = "Multi-Run Performance"

path = os.path.join(OUT_DIR,f'{name} Plots')
if not os.path.exists(path):
        os.makedirs(path)

joint_name = "64_joint_0_post"

# Replace 'path_to_csv' with the actual path to your CSV file.
csv_file_path = f'/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-02-02 Source Dynamics/Electrode Comparison Data/64/joint_0_post.csv'

df = pd.read_csv(csv_file_path)  # Read the dataset from the CSV file

df = df.drop(df.index[0])  # Remove the first row from the dataframe

df = df.iloc[:, :-4]  # remove the last four columns (unnecessary)

# Correct the column names based on your provided code
df.columns = ['source', 'precision', 'precision_std', 'recall', 'recall_std', 'f1-score', 'f1-score_std']

# Ensure all metric values are numeric
df[['precision', 'precision_std', 'recall', 'recall_std', 'f1-score', 'f1-score_std']] = \
    df[['precision', 'precision_std', 'recall', 'recall_std', 'f1-score', 'f1-score_std']].apply(pd.to_numeric)

# Plotting the metrics with error bars for each source
metrics = ['precision', 'recall', 'f1-score']
error_metrics = ['precision_std', 'recall_std', 'f1-score_std']

# Set the 'source' column as the index
df.set_index('source', inplace=True)

# Convert the source index to string to avoid any numeric sorting
df.index = df.index.map(str)

# Create a figure with subplots
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))

# Define colors for each metric
colors = ['blue', 'green', 'red']

# Plot each metric with error bars
for i, (metric, error_metric, color) in enumerate(zip(metrics, error_metrics, colors)):
    axs[i].bar(df.index, df[metric], yerr=df[error_metric], capsize=5, color=color)
    axs[i].set_title(metric.capitalize())
    axs[i].set_xlabel('Source')
    axs[i].set_ylabel('Value')
    axs[i].set_xticklabels(df.index, rotation=45)  # Rotate x-tick labels for better readability
    axs[i].set_yscale('linear')  # Explicitly set the y-axis scale to linear
    axs[i].set_ylim([0, 1])  # Set the y-axis limits from 0 to 1

fig.suptitle('Model Performance, 64 Stimulation Electrodes (Source Off, Post-Ablation)')

fig.savefig(path + f"/{joint_name}.png", dpi=300)

# Adjust the layout
plt.tight_layout()

# Show the plots
# plt.show()

fig, axs = plt.subplots(3, 1, figsize=(14,10))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure
axs[0].imshow(plt.imread(path + "/16_joint_0_post.png"), aspect='auto')
axs[1].imshow(plt.imread(path + "/32_joint_0_post.png"), aspect='auto')
axs[2].imshow(plt.imread(path + "/64_joint_0_post.png"), aspect='auto')
# fig.suptitle(f"{name}", fontsize = 10)
fig.savefig(path + "/Source Off.png", dpi = 300)
plt.tight_layout()

fig, axs = plt.subplots(3, 1, figsize=(14,10))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure
axs[0].imshow(plt.imread(path + "/16_joint_1_post.png"), aspect='auto')
axs[1].imshow(plt.imread(path + "/32_joint_1_post.png"), aspect='auto')
axs[2].imshow(plt.imread(path + "/64_joint_1_post.png"), aspect='auto')
# fig.suptitle(f"{name}", fontsize = 10)
fig.savefig(path + "/Source On.png", dpi = 300)
plt.tight_layout()