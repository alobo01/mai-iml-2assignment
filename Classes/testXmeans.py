import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from Classes.XMeans import XMeansAlgorithm

# Generate sample data with 3 clusters
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# Fit XMeans algorithm
xmeans = XMeansAlgorithm(initial_k=2, max_k=8, distance_metric='euclidean', max_iter=100)
predicted_labels, k = xmeans.fit(X)

# Evaluate accuracy using Adjusted Rand Index
ari = adjusted_rand_score(y, predicted_labels)
print(f"Adjusted Rand Index: {ari}")

# Plot the actual clusters
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.5)
plt.title("Actual Clusters")

# Plot the predicted clusters
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', s=50, alpha=0.5)
plt.title("Predicted Clusters")

plt.show()
