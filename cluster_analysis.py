import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Simulating 2D Particle Configurations
np.random.seed(42)
particles = np.random.rand(150, 2) 

# Identifying structural patterns using K-means
kmeans = KMeans(n_clusters=3, random_state=0).fit(particles)
labels = kmeans.labels_

# Plotting the results
plt.figure(figsize=(8,6))
plt.scatter(particles[:, 0], particles[:, 1], c=labels, cmap='plasma', edgecolors='k')
plt.title('Particle Configuration Clustering (ML Analysis)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()
