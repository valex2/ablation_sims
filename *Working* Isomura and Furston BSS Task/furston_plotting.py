import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ
OUT_DIR = "/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-01-26 Source Separation" #NOTE: change this for output folder
os.system('mkdir {}'.format(os.path.join(OUT_DIR,'plots')))

name = "Saving Figures"

path = os.path.join(OUT_DIR,f'{name} plots')
if not os.path.exists(path):
        os.makedirs(path)

# Read the CSV file into a DataFrame
df = pd.read_csv('/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-01-26 Source Separation/furston_classification_post/performance_off_post.csv', delimiter='\,')

# Group the data by source
grouped = df.groupby('source')

# Plotting each metric as a separate bar plot
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    plt.figure()
    for name, group in grouped:
        plt.bar(group['source'], group[metric], label=name)
    plt.xlabel('Source')
    plt.ylabel(metric)
    plt.legend()
    plt.title(f'model $\mathbf{{{metric}}}$ when source is off, post-ablation')
    plt.savefig(path + f"/{metric}_0_post.png", dpi=300)


# Read the CSV file into a DataFrame
df = pd.read_csv('/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-01-26 Source Separation/furston_classification_post/performance_on_post.csv', delimiter='\,')

# Group the data by source
grouped = df.groupby('source')

# Plotting each metric as a separate bar plot
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    plt.figure()
    for name, group in grouped:
        plt.bar(group['source'], group[metric], label=name)
    plt.xlabel('Source')
    plt.ylabel(metric)
    plt.legend()
    plt.title(f'model $\mathbf{{{metric}}}$ when source is on, post-ablation')
    plt.savefig(path + f"/{metric}_1_post.png", dpi=300)

# Read the CSV file into a DataFrame
df = pd.read_csv('/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-01-26 Source Separation/furston_classification_pre/performance_off_pre.csv', delimiter='\,')

# Group the data by source
grouped = df.groupby('source')

# Plotting each metric as a separate bar plot
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    plt.figure()
    for name, group in grouped:
        plt.bar(group['source'], group[metric], label=name)
    plt.xlabel('Source')
    plt.ylabel(metric)
    plt.legend()
    plt.title(f'model $\mathbf{{{metric}}}$ when source is off, pre-ablation')
    plt.savefig(path + f"/{metric}_0_pre.png", dpi=300)

# Read the CSV file into a DataFrame
df = pd.read_csv('/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-01-26 Source Separation/furston_classification_pre/performance_on_pre.csv', delimiter='\,')

# Group the data by source
grouped = df.groupby('source')

# Plotting each metric as a separate bar plot
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    plt.figure()
    for name, group in grouped:
        plt.bar(group['source'], group[metric], label=name)
    plt.xlabel('Source')
    plt.ylabel(metric)
    plt.legend()
    plt.title(f'model $\mathbf{{{metric}}}$ when source is on, pre-ablation')
    plt.savefig(path + f"/{metric}_1_pre.png", dpi=300)

# ----------------------------------------------------------------------------------------------- #

fig, axs = plt.subplots(3, 3, figsize=(18,10))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure
axs[0].imshow(plt.imread(path + "/precision_0_pre.png"), aspect='auto')
axs[1].imshow(plt.imread(path + "/recall_0_pre.png"), aspect='auto')
axs[2].imshow(plt.imread(path + "/f1-score_0_pre.png"), aspect='auto')

axs[3].imshow(plt.imread(path + "/precision_0_post.png"), aspect='auto')
axs[4].imshow(plt.imread(path + "/recall_0_post.png"), aspect='auto')
axs[5].imshow(plt.imread(path + "/f1-score_0_post.png"), aspect='auto')

axs[6].imshow(plt.imread(path + "/precision_1_post.png"), aspect='auto')
axs[7].imshow(plt.imread(path + "/recall_1_post.png"), aspect='auto')
axs[8].imshow(plt.imread(path + "/f1-score_1_post.png"), aspect='auto')

fig.suptitle(f'source classification performance after 70% ablation, balanced e/i network', fontsize = 10)
# fig.suptitle(f'{name}, pre-ablation -- MSE = {pre_mse:.3f}, CorrelationCoeff = {pre_correlation_coefficient:.3f}', fontsize = 10)
fig.savefig(path + "/overview_fig", dpi = 300)
plt.tight_layout()

# ----------------------------------------------------------------------------------------------- #

##### NOTE: display/close functionality for plotting ease #####
plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run

# ----------------------------------------------------------------------------------------------- # 