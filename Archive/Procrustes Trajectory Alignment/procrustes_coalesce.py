import os
import pandas as pd

#folders_of_interest = ['pre_procrustes', 'post_procrustes']
folders_of_interest = ['iz_null', 'iz_joint']

for folder in folders_of_interest:
    # Directory containing the CSV files
    OUT_DIR = os.path.join(os.environ['DM_T'])
    path = os.path.join(OUT_DIR,'vassilis_out')
    specific_path = os.path.join(path, folder)

    # Get a list of CSV file paths in the directory
    csv_files = [f for f in os.listdir(specific_path) if f.endswith('.csv')]
    csv_files.sort()  # Sort the list of files

    combined_df = None # Initialize an empty DataFrame to store the combined data

    # Iterate through the sorted CSV files
    for csv_file in csv_files:
        file_path = os.path.join(specific_path, csv_file)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Merge the data into the combined DataFrame
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df])

    output_csv_path = os.path.join(specific_path,f"combined_{folder}.csv")
    # Write the combined data to a single CSV file
    combined_df.to_csv(output_csv_path, index=False)

    print(f'CSV files in {specific_path} have been sorted and combined into {output_csv_path}.')
