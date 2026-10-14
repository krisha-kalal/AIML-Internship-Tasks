import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate synthetic 2D data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Scale features (Important: K-Means relies heavily on Euclidean distances)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and fit the KMeans model
# We define 4 clusters matching our generated data centers
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Extract cluster centroids
centroids = kmeans.cluster_centers_

# Visualize the results
plt.figure(figsize=(8, 6))
# Plot data points colored by their cluster assignment
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis', label='Data Points')
# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')

plt.title('K-Means Clustering (scikit-learn)')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.legend()
plt.show()
