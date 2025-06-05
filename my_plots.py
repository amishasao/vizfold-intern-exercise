import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('vizfold-intern-exercise\\sample_data.pickle', 'rb') as f:
    data = pickle.load(f)

single_plot = data['representations']['single']  # shape: (L, 256)
pair_plot = data['representations']['pair']  # shape: (L, 256, 256)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(single_plot[:, 0])
axs[0, 0].set_title("Single Representation: Feature 0")
axs[0, 0].set_xlabel("Residue index")
axs[0, 0].set_ylabel("Feature value")

axs[0, 1].matshow(single_plot[:, :10], aspect='auto', cmap='viridis')
axs[0, 1].set_title("Single Representation: First 10 Features")
axs[0, 1].set_xlabel("Residue index")
axs[0, 1].set_ylabel("Feature index")

axs[1, 0].matshow(pair_plot[:, :, 0], cmap='coolwarm')
axs[1, 0].set_title("Pair Representation: Feature 0")
axs[1, 0].set_xlabel("Residue index")
axs[1, 0].set_ylabel("Residue index")

axs[1, 1].matshow(pair_plot[:, :, 1], cmap='coolwarm')
axs[1, 1].set_title("Pair Representation: First 10 Features")
axs[1, 1].set_xlabel("Residue index")
axs[1, 1].set_ylabel("Residue index")

plt.tight_layout()
plt.show()
