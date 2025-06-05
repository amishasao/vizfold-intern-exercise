import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

with open('vizfold-intern-exercise/sample_data.pickle', 'rb') as f:
    data = pickle.load(f)

# we need all the representations to be of the same size
single_plot = data['representations']['single']  # shape: (L, 256)
pair_plot = data['representations']['pair']  # shape: (L, 128, 128)

# run t-SNE on the single representation
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
single_tsne = tsne.fit_transform(single_plot)

# run t-SNE on the pair representation
# flatten the pair representation
pair_flat = pair_plot.reshape(pair_plot.shape[0], -1)
pair_tsne = tsne.fit_transform(pair_flat)

# Plotting the t-SNE results
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].scatter(single_tsne[:, 0], single_tsne[:, 1], alpha=0.5)
axs[0].set_title("t-SNE of Single Representation")
axs[0].set_xlabel("t-SNE Component 1")
axs[0].set_ylabel("t-SNE Component 2")
axs[1].scatter(pair_tsne[:, 0], pair_tsne[:, 1], alpha=0.5)
axs[1].set_title("t-SNE of Pair Representation")
axs[1].set_xlabel("t-SNE Component 1")
axs[1].set_ylabel("t-SNE Component 2")
plt.tight_layout()
plt.show()
