# Basic usage
from weighted_clustering import DataMatrix, WeightedClustering

# Load your data
data = DataMatrix("your_economic_data.csv")

# Initialize and fit clustering
clusterer = WeightedClustering(n_clusters=3, beta=1.5)
clusterer.fit(data)

# Get results
frequencies = clusterer.get_cluster_frequencies()
print(f"Cluster distribution: {frequencies}")