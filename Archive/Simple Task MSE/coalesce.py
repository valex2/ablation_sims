import os
import pandas as pd

filetype_of_interest = 'povs/povs_post'

# Directory containing the CSV files
OUT_DIR = os.path.join(os.environ['DM_T'])
path = os.path.join(OUT_DIR,'vassilis_out')
specific_path = os.path.join(path, filetype_of_interest)

# Get a list of CSV file paths in the directory
csv_files = [file for file in os.listdir(specific_path) if file.endswith('.csv')]
sorted_csv_files = [file for file in os.listdir(specific_path) if file.endswith('.csv')].sort()
print(f"csv files: {sorted_csv_files}")

# Create an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Loop through each CSV file and read its contents into a DataFrame
for csv_file in csv_files:
    file_path = os.path.join(specific_path, csv_file)
    column_name = csv_file.replace('.csv', '')  # Extract column name from file name
    
    # Read the CSV file into a DataFrame with a single column
    df = pd.read_csv(file_path, header=None, names=[column_name])
    
    # Append the DataFrame to the combined DataFrame
    combined_df = pd.concat([combined_df, df], axis=1)

# Save the combined DataFrame to a new CSV file
combined_csv_path = f'{specific_path}/combined.csv'
combined_df.to_csv(combined_csv_path, index=False)

print(f"Combined CSV saved at: {combined_csv_path}")
