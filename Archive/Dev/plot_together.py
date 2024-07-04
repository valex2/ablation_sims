import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of file paths to your PNG images
image_files = [
    '/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/Comparisons/Plots/null_mse.png',
    '/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/Comparisons/Plots/iz_null_mse.png',
    '/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/Comparisons/Plots/adaptive_lif_null.png',
    # '/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/Comparisons/Plots/bio_null.png',
    '/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/Comparisons/Plots/conn_mse.png',
    '/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/Comparisons/Plots/iz_conn_mse.png',
    '/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/Comparisons/Plots/adaptive_lif_conn.png',
    # '/Users/Vassilis/Desktop/BIL/valex_23_coupled_snn/Analysis/Comparisons/Plots/bio_conn.png'
]

# Create a 2x4 grid
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Loop through the image files and display them in the grid
for i, ax in enumerate(axes.ravel()):
    if i < len(image_files):
        img = mpimg.imread(image_files[i])
        ax.imshow(img)
    ax.axis('off')  # Turn off axes

# Adjust the layout
plt.tight_layout()

# Save the grid as a PNG
plt.savefig('image_grid.png', dpi=300, bbox_inches='tight')
plt.show()
