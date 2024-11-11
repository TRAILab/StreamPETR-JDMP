import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to the pickle file and output directory
file_path = 'output/uniad_anchors/motion_anchor_infos_mode6.pkl'  # Update with correct path if necessary
output_path = 'output/uniad_anchors/anchor_viz.png'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Simplified class names for each of the 4 sets
class_labels = ["vehicles", "bikes", "peds", "constr"]

try:
    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Extract 'anchors_all' from the dictionary
    anchors_all = data.get('anchors_all', None)
    
    # Check if 'anchors_all' exists and has the expected structure
    if anchors_all and isinstance(anchors_all, list) and len(anchors_all) == 4:
        # Define a list of colors for the 4 anchor sets
        colors = ['red', 'blue', 'green', 'orange']
        
        plt.figure(figsize=(12, 10))  # Create a figure for the plot
        
        for i, (anchors_array, label) in enumerate(zip(anchors_all, class_labels)):
            if isinstance(anchors_array, np.ndarray) and anchors_array.shape == (6, 12, 2):
                color = colors[i]  # Use a different color for each anchor set
                
                # Iterate over the (6, 12, 2) matrix
                for j in range(6):
                    points = anchors_array[j]
                    for k in range(12):
                        x_vals, y_vals = points[k]
                        plt.plot(
                            x_vals, y_vals, marker='o', color=color, label=label if j == 0 and k == 0 else ""
                        )
            else:
                print(f"Unexpected shape for anchor set {i}: {anchors_array.shape}")
        
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Anchor Visualizations with Class Labels")
        plt.legend(title="Anchor Classes", loc='upper right')
        plt.tight_layout()
        
        # Save the plot to the specified path
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")
    else:
        print("The 'anchors_all' entry is missing or does not have the expected structure.")
except FileNotFoundError:
    print("The file was not found. Please check the file path.")
except pickle.UnpicklingError:
    print("The file could not be loaded. It might not be a valid pickle file.")
except Exception as e:
    print(f"An error occurred: {e}")
