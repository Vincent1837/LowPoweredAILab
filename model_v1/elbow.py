import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt
import pickle

# Assuming `feat_score` is the DFT loss array from your feature selection function
def find_elbow_point(feat_score):
    # Sort the features by their score
    sorted_scores = np.sort(feat_score)
    # The index of the scores, sorted
    sorted_idx = np.argsort(feat_score)
    
    # Use the KneeLocator to find the elbow
    kneedle = KneeLocator(range(len(sorted_scores)), sorted_scores, curve='convex', direction='decreasing')
    
    elbow_point = kneedle.elbow
    return elbow_point, sorted_idx

# Example usage in your code
selected_idx = pickle.load(open("dft_selected.pickle", "rb"))
feat_score = pickle.load(open("dft_loss.pickle", "rb"))
elbow_point, sorted_idx = find_elbow_point(feat_score)
print(f"Elbow point at index: {elbow_point}")

# Adjust the threshold based on the elbow point
selected_idx = sorted_idx[:elbow_point]

def visualize_dft_loss(feat_score, selected_idx):
    plt.figure(figsize=(10, 6))
    
    # Plot all feature losses
    plt.plot(range(len(feat_score)), np.sort(feat_score), label="DFT Loss")
    
    # Highlight the selected features
    plt.scatter(selected_idx, np.sort(feat_score)[selected_idx], color='red', label="Selected Features")
    
    plt.xlabel("Feature Index")
    plt.ylabel("DFT Loss")
    plt.title("DFT Loss vs. Feature Index")
    plt.legend()
    plt.show()

# Example usage
visualize_dft_loss(feat_score, selected_idx)