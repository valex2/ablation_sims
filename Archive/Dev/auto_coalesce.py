import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fractions import Fraction
import matplotlib.cm as cm

folders_of_interest = ['bio_uncon_null', 'bio_uncon_joint']

for folder in folders_of_interest:
        OUT_DIR = os.path.join(os.environ['DM_T'])
        path = os.path.join(OUT_DIR,'vassilis_out')
        specific_path = os.path.join(path, folder)

        # Get a list of CSV file paths in the directory
        csv_files = [f for f in os.listdir(specific_path) if f.endswith('.csv')]
        csv_files.sort()  # Sort the list of files

        df = pd.read_csv(os.path.join(specific_path, csv_files[0]))
        x_vars = [number for number in df.columns.to_list()]

        ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for ratio in ratios:
            new_df = pd.DataFrame(columns=x_vars)

            for file in csv_files:
                if float(file.split('_')[4].replace('.csv','')) == ratio:
                    csv_path = os.path.join(specific_path, file) 
                    df = pd.read_csv(csv_path)
                    new_df = pd.concat([new_df, df], ignore_index=True)

            coalesced_path = os.path.join(path, 'coalesced')  # Main directory
            if not os.path.exists(coalesced_path):
                os.makedirs(coalesced_path)

            combined_csv_path = os.path.join(coalesced_path, f"{ratio}.csv")
            new_df.to_csv(combined_csv_path, index=False)
            print(f"Combined CSV saved at: {combined_csv_path}")