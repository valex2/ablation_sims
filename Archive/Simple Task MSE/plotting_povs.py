import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plot_until = 5

csv_file_path = '/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/EI Sweep Data/combined.csv' # Path to the CSV file
df = pd.read_csv(csv_file_path) # Read the CSV file into a DataFrame

def get_num_from_string(s): # Sort the list using the custom sorting key function
    return int(s.split("_")[1])  # Assuming the num is always the second part after splitting by "_"
sorted_column_names = sorted(df.columns, key=get_num_from_string)
df = df[sorted_column_names]
print(df.columns)

n = len(df.iloc[:, 0]) # pull the number of points in each column
n = np.arange(n) # make a linear map of the data points and the columns

colormap = cm.get_cmap('RdYlGn', len(df.columns)) # Get a colormap ranging from red to green (reversed RdYlGn colormap)
reversed_colormap = colormap.reversed()

for i, col in enumerate(df.columns): # Plot each column with color based on column index
    color = reversed_colormap(i)
    if i == 0 or i == len(df.columns) - 1:  # Plot only the first and last columns with legend labels
        plt.plot(n[:plot_until], df[col][:plot_until], color=color, label=f'{float(col.split("_")[3])*10}:1 ablation')
    else:
        plt.plot(n[:plot_until], df[col][:plot_until], color=color)

plt.xlabel('PC')  # Change x-axis label
plt.ylabel('Percent of Variance Explained')  # Change y-axis label
plt.title('POV, sweeping across ablation')
plt.legend()

plt.xticks(np.arange(0, plot_until), np.arange(1, plot_until+1))  # Show only integers in tickmarks
plt.yticks(np.arange(0, 101, 10))  # Set y-axis ticks from 0 to 100 with step of 10
plt.ylim(0, 100)  # Set y-axis range from 0 to 100

plt.show()
