import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fractions import Fraction
import matplotlib.cm as cm

folder_path = '/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/Comparisons/Bio/bio_playing_joint'

csv_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.csv')])

ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
new_files = []

df = pd.read_csv(os.path.join(folder_path, csv_files[0]))
x_vars = [number for number in df.columns.to_list()]

for ratio in ratios:
    new_df = pd.DataFrame(columns=x_vars)

    for file in csv_files:
        if float(file.split('_')[4].replace('.csv','')) == ratio:
            csv_path = os.path.join(folder_path, file) 
            df = pd.read_csv(csv_path)
            new_df = pd.concat([new_df, df], ignore_index=True)
    
    combined_csv_path = f'/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/Comparisons/Bio/bio_playing_joint/{ratio}.csv'
    new_df.to_csv(combined_csv_path, index=False)
    print(f"Combined CSV saved at: {combined_csv_path}")
