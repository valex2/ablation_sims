import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fractions import Fraction
import matplotlib.cm as cm

def fractions_to_ratios(fraction_list):
    ratio_strings = []
    num_neur = 100
    for fraction in fraction_list:
        excitatory = int(np.ceil(num_neur * fraction) / 10)
        inhibitory = int(np.ceil(num_neur * (1 - fraction)) / 10)
        ratio_strings.append(f"{excitatory}:{inhibitory}")
    return ratio_strings


folder_path = '/Users/Vassilis/Desktop/BIL/conn_mse_2'
name = 'Robustness of Sparsely Connected Network'
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
csv_files = sorted(csv_files)
legend_names = [comp.split('_')[4] for comp in csv_files]
legend_names = [float(legend.replace('.csv','')) for legend in legend_names]
legend_names = fractions_to_ratios(legend_names)

plt.figure(figsize=(10, 6))

x_vars = []
for i, csv_file in enumerate(csv_files):
    csv_path = os.path.join(folder_path, csv_file)
    
    df = pd.read_csv(csv_path)
    if len(x_vars) == 0:
        x_vars = [round(float(number), 2) for number in df.columns.to_list()]
    
    y_vals = df.iloc[0]

    color = cm.viridis(i / len(csv_files))  # Colormap for line colors
    
    plt.plot(x_vars, y_vals, label=f'{legend_names[i]}', color=color)

plt.xlabel('Lesioning Fraction')
plt.ylabel('Pre/Post Ablation MSE')
plt.title(name, fontsize=14, fontweight='bold')
plt.ylim((0, 0.3))
plt.legend(title='E:I')  # Set legend title
plt.grid()

folder_path = '/Users/Vassilis/Desktop/BIL/null_mse_2'
name = 'Robustness of Unconnected Network'
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
csv_files = sorted(csv_files)
legend_names = [int(np.ceil(float(comp.split('_')[4].replace('.csv',''))*100)) for comp in csv_files]
print(legend_names)

plt.figure(figsize=(10, 6))

x_vars = []
for i, csv_file in enumerate(csv_files):
    csv_path = os.path.join(folder_path, csv_file)
    
    df = pd.read_csv(csv_path)
    if len(x_vars) == 0:
        x_vars = [round(float(number), 2) for number in df.columns.to_list()]
    
    y_vals = df.iloc[0]

    color = cm.viridis(i / len(csv_files))  # Colormap for line colors
    
    plt.plot(x_vars, y_vals, label=f'{legend_names[i]}', color=color)

plt.xlabel('Lesioning Fraction')
plt.ylabel('Pre/Post Ablation MSE')
plt.title(name, fontsize=14, fontweight='bold')
plt.ylim((0, 0.3))
plt.legend(title='#E Neurons')  # Set legend title
plt.grid()


plt.show()