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
OUT_DIR = "/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability" #NOTE: change this for output folder
os.system('mkdir {}'.format(os.path.join(OUT_DIR,'plots')))

# azimuth = -95, elevation= 41

name = "joint_plots" # NOTE: change this for the name of the experiment
encoders = False # NOTE: change this for the type of experiment

path = os.path.join(OUT_DIR,f'{name}')
if not os.path.exists(path):
        os.makedirs(path)

# ------------------------------------------------------------------------------------#
# NOTE: this part of the code makes the final panel figure including all the plots
fig, axs = plt.subplots(2, 4, figsize=(18,7))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure

axs[0].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x4/classification_accuracy_together.png"), aspect='auto')
axs[1].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x12/classification_accuracy_together.png"), aspect='auto')
axs[2].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x20/classification_accuracy_together.png"), aspect='auto')
axs[3].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x28/classification_accuracy_together.png"), aspect='auto')

axs[4].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x32/classification_accuracy_together.png"), aspect='auto')
axs[5].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x40/classification_accuracy_together.png"), aspect='auto')
axs[6].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x52/classification_accuracy_together.png"), aspect='auto')
axs[7].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x64/classification_accuracy_together.png"), aspect='auto')

fig.suptitle(f"{name}", fontsize = 10)
fig.tight_layout()  # Reduce whitespace and make the layout tighter
fig.savefig(path + "/accuracy_panel.png", dpi = 300)

# ------------------------------------------------------------------------------------#
# NOTE: this part of the code makes the final panel figure including all the plots
fig, axs = plt.subplots(2, 4, figsize=(18,7))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure

axs[0].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x4/frobenius_std_together.png"), aspect='auto')
axs[1].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x12/frobenius_std_together.png"), aspect='auto')
axs[2].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x20/frobenius_std_together.png"), aspect='auto')
axs[3].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x28/frobenius_std_together.png"), aspect='auto')

axs[4].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x32/frobenius_std_together.png"), aspect='auto')
axs[5].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x40/frobenius_std_together.png"), aspect='auto')
axs[6].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x52/frobenius_std_together.png"), aspect='auto')
axs[7].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x64/frobenius_std_together.png"), aspect='auto')

fig.suptitle(f"{name}", fontsize = 10)
fig.tight_layout()  # Reduce whitespace and make the layout tighter
fig.savefig(path + "/frobenius_std_panel.png", dpi = 300)

# ------------------------------------------------------------------------------------#
# NOTE: this part of the code makes the final panel figure including all the plots
fig, axs = plt.subplots(2, 4, figsize=(18,7))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure

axs[0].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x4/frobenius_averages_together.png"), aspect='auto')
axs[1].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x12/frobenius_averages_together.png"), aspect='auto')
axs[2].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x20/frobenius_averages_together.png"), aspect='auto')
axs[3].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x28/frobenius_averages_together.png"), aspect='auto')

axs[4].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x32/frobenius_averages_together.png"), aspect='auto')
axs[5].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x40/frobenius_averages_together.png"), aspect='auto')
axs[6].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x52/frobenius_averages_together.png"), aspect='auto')
axs[7].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x64/frobenius_averages_together.png"), aspect='auto')

fig.suptitle(f"{name}", fontsize = 10)
fig.tight_layout()  # Reduce whitespace and make the layout tighter
fig.savefig(path + "/frobenius_averages_panel.png", dpi = 300)

# ------------------------------------------------------------------------------------#
# NOTE: this part of the code makes the final panel figure including all the plots
fig, axs = plt.subplots(2, 4, figsize=(18,7))
axs = axs.flatten()
for ax in axs: ax.axis("off") # remove the markings from the original figure

axs[0].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x4/pre - post, entire pop, FDD_rotated.png"), aspect='auto')
axs[1].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x12/pre - post, entire pop, FDD_rotated.png"), aspect='auto')
axs[2].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x20/pre - post, entire pop, FDD_rotated.png"), aspect='auto')
axs[3].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x28/pre - post, entire pop, FDD_rotated.png"), aspect='auto')

axs[4].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x32/pre - post, entire pop, FDD_rotated.png"), aspect='auto')
axs[5].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x40/pre - post, entire pop, FDD_rotated.png"), aspect='auto')
axs[6].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x52/pre - post, entire pop, FDD_rotated.png"), aspect='auto')
axs[7].imshow(plt.imread("/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-03-15 Weight Scaling + Electrode Variability/4x64/pre - post, entire pop, FDD_rotated.png"), aspect='auto')

fig.suptitle(f"{name}", fontsize = 10)
fig.tight_layout()  # Reduce whitespace and make the layout tighter
fig.savefig(path + "/pre - post, entire pop, FDD_rotated_panel.png", dpi = 300)
# ----------------------------------------------------------------------------------------------- #
##### NOTE: display/close functionality for plotting ease #####
plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run
# ----------------------------------------------------------------------------------------------- # 
