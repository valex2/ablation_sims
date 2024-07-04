import pandas as pd
import numpy as np
import os

# Sample list of file paths for demonstration purposes
# In practice, replace these with the actual file paths
csv_file_paths = []
for i in range(40):
    file_path = f"/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-02-02 Source Dynamics/Electrode Comparison Data/64/furston_classification_post_electrodes=64/Off/performance_off_post_{i}.csv"
    if os.path.exists(file_path):
        csv_file_paths.append(file_path)

# Function to read and concatenate CSV files into one DataFrame
def concatenate_csv(files):
    dfs = [pd.read_csv(file) for file in files]
    concatenated_df = pd.concat(dfs, ignore_index=True)
    return concatenated_df

# Concatenate all CSV files into one DataFrame
concatenated_df = concatenate_csv(csv_file_paths)

# Select only numeric columns for aggregation
numeric_cols = concatenated_df.select_dtypes(include=[np.number]).columns.tolist()

# Group by 'source' and calculate mean and standard deviation only for numeric columns
aggregated_df = concatenated_df.groupby('source')[numeric_cols].agg(['mean', 'std']).reset_index()

# Save the aggregated DataFrame to a new CSV file
output_file_path = f"/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/24-02-02 Source Dynamics/Electrode Comparison Data/64/joint_0_post.csv"
aggregated_df.to_csv(output_file_path, index=False)
