import pickle
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

with open('6KWC\\test_5667f_all_rank_001_alphafold2_ptm_model_2_seed_000.pickle', 'rb') as f:
    data = pickle.load(f)

single_plot = data['representations']['single']  # shape: (L, 256)
pair_plot = data['representations']['pair']  # shape: (L, 128, 128)

reducer = umap.UMAP(n_components=2, random_state=42)

scaled_data = StandardScaler().fit_transform(single_plot)
single_umap = reducer.fit_transform(scaled_data)
pair_flat = pair_plot.reshape(pair_plot.shape[0], -1)
pair_umap = reducer.fit_transform(pair_flat)

# Plotting the UMAP results
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].scatter(single_umap[:, 0], single_umap[:, 1], alpha=0.5)
axs[0].set_title("UMAP of Single Representation")
axs[0].set_xlabel("UMAP Component 1")
axs[0].set_ylabel("UMAP Component 2")
axs[1].scatter(pair_umap[:, 0], pair_umap[:, 1], alpha=0.5)
axs[1].set_title("UMAP of Pair Representation")
axs[1].set_xlabel("UMAP Component 1")
axs[1].set_ylabel("UMAP Component 2")
plt.tight_layout()
plt.show()
