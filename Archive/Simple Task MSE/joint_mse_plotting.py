import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fractions import Fraction
import matplotlib.cm as cm
from scipy.optimize import curve_fit

def fractions_to_ratios(fraction_list):
    ratio_strings = []
    num_neur = 100
    for fraction in fraction_list:
        excitatory = int(np.ceil(num_neur * fraction) / 10)
        inhibitory = int(np.ceil(num_neur * (1 - fraction)) / 10)
        ratio_strings.append(f"{excitatory}:{inhibitory}")
    return ratio_strings

def func(x, a, b):
    return a * np.exp(b*x)

folder_path = '/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/Comparisons/Bio/bio_playing_joint/Combined'
name = 'Robustness of Bio Sparsely Connected Network'
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
csv_files = sorted(csv_files, reverse=True)
legend_names = [f"{(int(np.ceil(float(comp.replace('.csv',''))*100)))}:{100 - (int(np.ceil(float(comp.replace('.csv',''))*100)))}" for comp in csv_files]

plt.figure(figsize=(10, 6))

x_vars = []
for i, csv_file in enumerate(csv_files):
    csv_path = os.path.join(folder_path, csv_file)
    
    df = pd.read_csv(csv_path)
    if len(x_vars) == 0:
        x_vars = [round(float(number), 2) for number in df.columns.to_list()]
    
    y_vals = df.iloc[0:]
    y_errors = np.std(y_vals, axis=0)
    y_vals = np.mean(y_vals, axis=0)

    x_vars = np.array([float(x_var) for x_var in x_vars])
    y_vals = np.array([float(y_val) for y_val in y_vals])
    
    color = cm.viridis(i / len(csv_files))  # Colormap for line colors
    
    #plt.plot(x_vars, y_vals, label=f'{legend_names[i]}', color=color)
    params, covar = curve_fit(func, x_vars, y_vals, maxfev=5000)
    cov_diag = np.diag(covar)

    a, b = params
    y_pred = func(x_vars, a, b)
    plt.plot(x_vars*100, y_pred, color=color, label=f'{legend_names[i]}')
    plt.fill_between(x_vars*100, y_pred - np.sqrt(cov_diag[0]), y_pred + np.sqrt(cov_diag[0]), color=color, alpha=0.1)
    #plt.fill_between(x_vars, y_vals - y_errors, y_vals + y_errors, color=color, alpha=0.1)

plt.xlabel('Lesioning Precentage')
plt.ylabel('Pre/Post Ablation MSE')
plt.title(name, fontsize=14, fontweight='bold')
plt.ylim((0, 0.2))
plt.legend(title='E:I')  # Set legend title
plt.grid()
plt.savefig(folder_path + "/conn_mse", dpi = 300)

folder_path = '/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/Comparisons/Bio/bio__playing_null/Combined'
name = 'Robustness of Bio Excitatory Network, Controlling For Population Size'
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
csv_files = sorted(csv_files, reverse=True)
legend_names = [f"{(int(np.ceil(float(comp.replace('.csv',''))*100)))}" for comp in csv_files]

plt.figure(figsize=(10, 6))

x_vars = []
for i, csv_file in enumerate(csv_files):
    csv_path = os.path.join(folder_path, csv_file)
    
    df = pd.read_csv(csv_path)
    if len(x_vars) == 0:
        x_vars = [round(float(number), 2) for number in df.columns.to_list()]
    
    y_vals = df.iloc[0:]
    y_errors = np.std(y_vals, axis=0)
    y_vals = np.mean(y_vals, axis=0)

    x_vars = np.array([float(x_var) for x_var in x_vars])
    y_vals = np.array([float(y_val) for y_val in y_vals])
    
    color = cm.viridis(i / len(csv_files))  # Colormap for line colors
    
    #plt.plot(x_vars, y_vals, label=f'{legend_names[i]}', color=color)
    params, covar = curve_fit(func, x_vars, y_vals, maxfev=5000)
    cov_diag = np.diag(covar)

    a, b = params
    y_pred = func(x_vars, a, b)
    plt.plot(x_vars*100, y_pred, color=color, label=f'{legend_names[i]}')
    plt.fill_between(x_vars*100, y_pred - np.sqrt(cov_diag[0]), y_pred + np.sqrt(cov_diag[0]), color=color, alpha=0.1)
    #plt.fill_between(x_vars, y_vals - y_errors, y_vals + y_errors, color=color, alpha=0.1)

plt.xlabel('Lesioning Precentage')
plt.ylabel('Pre/Post Ablation MSE')
plt.title(name, fontsize=14, fontweight='bold')
plt.ylim((0, 0.2))
plt.legend(title='#E Neurons')  # Set legend title
plt.grid()
plt.savefig(folder_path + "/null_mse", dpi = 300)


plt.show()