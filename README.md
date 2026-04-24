# Soft-Matter-Clustering-ML
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Simulating Particle Configurations (Random 2D coordinates)
np.random.seed(42)
particles = np.random.rand(100, 2) 

# 2. Applying K-means to identify structural patterns
kmeans = KMeans(n_clusters=3, random_state=0).fit(particles)
labels = kmeans.labels_

# 3. Visualization
plt.scatter(particles[:, 0], particles[:, 1], c=labels, cmap='viridis')
plt.title('Particle Configuration Clustering (Model System)')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.show()
