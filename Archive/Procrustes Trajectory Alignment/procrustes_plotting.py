import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load the "pre" and "post" datasets from CSV files
pre_df = pd.read_csv('/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/combined_pre_procrustes.csv')
post_df = pd.read_csv('/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/combined_post_procrustes.csv')

pre_df.sort_values(by=['ratio'], inplace=True)
post_df.sort_values(by=['ratio'], inplace=True)

# Extract data for plotting
pre_ratios = pre_df['ratio']
pre_averages = pre_df['average']
pre_stds = pre_df['std']

post_ratios = post_df['ratio']
post_averages = post_df['average']
post_stds = post_df['std']
avg_del_angle = post_df['avg_del_angle']

number_of_trials = 30

# Create a new figure for the plot
plt.figure(figsize=(10, 6))

# Plot the "pre" dataset with error bars
plt.errorbar(pre_ratios, pre_averages, yerr=pre_stds, fmt='o-', label='pre-ablation', capsize=5, color = 'blue')

# Plot the "post" dataset with error bars
plt.errorbar(post_ratios, post_averages, yerr=post_stds, fmt='o-', label='post-ablation', capsize=5, color = 'orange')

# for i in range(len(pre_ratios)):
#     t_statistic, p_value = stats.ttest_ind_from_stats(pre_averages[i], pre_stds[i], number_of_trials, post_averages[i], post_stds[i], number_of_trials)
#     plt.annotate(f'p-value: {p_value:.8f}', (pre_ratios[i], pre_averages[i] + 0.01), fontsize=10, ha='center', va='bottom')

# Customize the plot
plt.title('Trajectory Comparison Between Pre and Post-ablation Neural State Spaces')
plt.xlabel('E:I Ratio')
plt.ylabel('Procrustes Similarity, 25 trials at 70% ablation')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig('/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/29_ratio_plot.png', dpi=300)

plt.figure(figsize=(10, 6))
plt.plot(pre_ratios, avg_del_angle, label='avg_del_angle', color = 'brown')
plt.title('Average Angle Change Between Pre and Post-ablation Neural State Spaces')
plt.xlabel('E:I Ratio')
plt.ylabel('Average Angle Change, Degrees, 25 trials at 70% ablation')
plt.grid(True)
plt.savefig('/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/29_angle_plot.png', dpi=300)


plt.show()