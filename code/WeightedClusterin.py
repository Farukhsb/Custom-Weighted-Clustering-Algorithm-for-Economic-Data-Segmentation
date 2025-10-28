"""
Custom Weighted Clustering Algorithm for Economic Data Segmentation

A sophisticated implementation of adaptive weighted clustering with dynamic feature
weighting and robust centroid updates. Designed for economic and policy data analysis.

Author: Abdullah Faruk
Date: 2024
"""

import numpy as np
from typing import List, Tuple, Optional, Union


class DataMatrix:
    """
    Enhanced matrix class for handling economic data with standardization
    and distance calculations.
    """
    
    def __init__(self, data_source: Union[str, np.ndarray]):
        """
        Initialize DataMatrix from file path or numpy array.
        
        Args:
            data_source: Path to CSV file or numpy array containing the data
        """
        if isinstance(data_source, str):
            self.data = np.loadtxt(open(data_source), delimiter=",")
        elif isinstance(data_source, np.ndarray):
            self.data = data_source
        else:
            raise ValueError("data_source must be string (file path) or numpy array")
            
        self.n_samples, self.n_features = self.data.shape
    
    def standardize(self) -> None:
        """
        Standardize each column to [0, 1] range with zero-division protection.
        Constant columns are set to zero.
        """
        for j in range(self.n_features):
            column = self.data[:, j]
            min_val, max_val = np.amin(column), np.amax(column)
            col_range = max_val - min_val
            
            if col_range == 0:
                # Handle constant columns
                self.data[:, j] = 0
            else:
                self.data[:, j] = (column - min_val) / col_range
    
    def calculate_feature_spread(self, other_point: np.ndarray, 
                               weights: np.ndarray, beta: float, 
                               reference_idx: int = 0) -> float:
        """
        Calculate weighted spread/distance between a reference point and another point.
        
        Args:
            other_point: Point to compare against
            weights: Feature weights array
            beta: Distance exponent parameter
            reference_idx: Index of reference point in current matrix
            
        Returns:
            Weighted distance value
        """
        spread = np.absolute(self.data[reference_idx] - other_point)
        spread = np.power(spread, beta)
        weighted_weights = np.power(weights, beta)
        weighted_spread = np.multiply(spread, weighted_weights)
        return np.sum(weighted_spread)
    
    def compute_distances(self, other_matrix: 'DataMatrix', 
                         weights: np.ndarray, beta: float, 
                         reference_idx: int = 0) -> 'DataMatrix':
        """
        Compute weighted distances between reference point and all points in other matrix.
        
        Args:
            other_matrix: Matrix to compare against
            weights: Feature weights array
            beta: Distance exponent parameter
            reference_idx: Index of reference point
            
        Returns:
            DataMatrix containing computed distances
        """
        distances = np.array([
            self.calculate_feature_spread(point, weights, beta, reference_idx) 
            for point in other_matrix.data
        ])
        return DataMatrix(distances)
    
    def get_frequency_distribution(self) -> List[List[float]]:
        """
        Get frequency count of unique values in the matrix.
        
        Returns:
            List of [value, count] pairs
        """
        unique_values, counts = np.unique(self.data, return_counts=True)
        result = np.column_stack((unique_values, counts))
        return result.tolist()


class WeightedClustering:
    """
    Custom weighted clustering algorithm with adaptive feature weighting
    and robust centroid updates.
    """
    
    def __init__(self, n_clusters: int, beta: float = 1.5, max_iterations: int = 100):
        """
        Initialize clustering algorithm.
        
        Args:
            n_clusters: Number of clusters to form
            beta: Weight adaptation parameter (beta > 1)
            max_iterations: Maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.beta = beta
        self.max_iterations = max_iterations
        self.weights = None
        self.centroids = None
        self.labels = None
        
        if beta <= 1:
            raise ValueError("Beta parameter must be greater than 1")
    
    def _initialize_weights(self, n_features: int) -> np.ndarray:
        """
        Initialize feature weights using uniform distribution.
        
        Args:
            n_features: Number of features in the dataset
            
        Returns:
            Array of initial feature weights
        """
        random_weights = np.random.uniform(0.5, 1.0, n_features)
        # Normalize so average weight is 0.5
        scaler = np.sum(random_weights) * 2 / n_features
        return random_weights / scaler
    
    def _calculate_dispersion(self, data: DataMatrix, centroids: DataMatrix,
                            labels: np.ndarray, feature_idx: int) -> float:
        """
        Calculate dispersion for a specific feature across all clusters.
        
        Args:
            data: Input data matrix
            centroids: Current cluster centroids
            labels: Current cluster assignments
            feature_idx: Index of feature to calculate dispersion for
            
        Returns:
            Dispersion value for the feature
        """
        dispersion = 0.0
        n_samples = data.n_samples
        
        for cluster_idx in range(self.n_clusters):
            cluster_mask = (labels == cluster_idx + 1)
            cluster_data = data.data[cluster_mask, feature_idx]
            centroid_value = centroids.data[cluster_idx, feature_idx]
            
            if len(cluster_data) > 0:
                dispersion += np.sum(np.abs(cluster_data - centroid_value) ** self.beta)
                
        return dispersion
    
    def _update_weights(self, data: DataMatrix, centroids: DataMatrix, 
                       labels: np.ndarray) -> np.ndarray:
        """
        Update feature weights based on current cluster dispersion.
        
        Args:
            data: Input data matrix
            centroids: Current cluster centroids
            labels: Current cluster assignments
            
        Returns:
            Updated feature weights array
        """
        n_features = data.n_features
        dispersions = np.zeros(n_features)
        new_weights = np.zeros(n_features)
        
        # Calculate dispersion for each feature
        for j in range(n_features):
            dispersions[j] = self._calculate_dispersion(data, centroids, labels, j)
        
        # Handle zero dispersions (perfect clustering for that feature)
        zero_dispersion_mask = (dispersions == 0)
        if np.any(zero_dispersion_mask):
            new_weights[zero_dispersion_mask] = 0
        
        # Calculate normalization factor for non-zero dispersions
        non_zero_mask = ~zero_dispersion_mask
        if np.any(non_zero_mask):
            inverse_dispersions = 1.0 / (dispersions[non_zero_mask] ** (1.0 / (self.beta - 1)))
            total_inverse = np.sum(inverse_dispersions)
            
            if total_inverse > 0:
                new_weights[non_zero_mask] = 1.0 / (inverse_dispersions * total_inverse)
        
        return new_weights
    
    def _calculate_centroids(self, data: DataMatrix, labels: np.ndarray) -> DataMatrix:
        """
        Calculate new centroids using robust mean-median combination.
        
        Args:
            data: Input data matrix
            labels: Current cluster assignments
            
        Returns:
            DataMatrix containing new centroids
        """
        centroids_list = []
        
        for cluster_idx in range(1, self.n_clusters + 1):
            cluster_mask = (labels == cluster_idx)
            cluster_data = data.data[cluster_mask]
            
            if len(cluster_data) == 0:
                # If cluster is empty, reinitialize randomly
                random_point = data.data[np.random.randint(0, data.n_samples)]
                centroids_list.append(random_point)
            else:
                # Robust centroid: average of mean and median
                cluster_centroid = []
                for feature_idx in range(data.n_features):
                    feature_values = cluster_data[:, feature_idx]
                    robust_center = (np.mean(feature_values) + np.median(feature_values)) / 2
                    cluster_centroid.append(robust_center)
                centroids_list.append(cluster_centroid)
        
        return DataMatrix(np.array(centroids_list))
    
    def fit(self, data: DataMatrix) -> None:
        """
        Perform weighted clustering on the input data.
        
        Args:
            data: Input data matrix (will be standardized)
        """
        # Standardize data
        data.standardize()
        
        n_samples = data.n_samples
        n_features = data.n_features
        
        # Initialize weights and labels
        self.weights = self._initialize_weights(n_features)
        self.labels = np.zeros(n_samples, dtype=int)
        
        # Initialize centroids with random points from data
        random_indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        self.centroids = DataMatrix(data.data[random_indices])
        
        previous_labels = np.copy(self.labels)
        
        # Main clustering loop
        for iteration in range(self.max_iterations):
            # Assign points to nearest centroids
            for i in range(n_samples):
                distances = data.compute_distances(self.centroids, self.weights, self.beta, i)
                self.labels[i] = np.argmin(distances.data) + 1
            
            # Check for convergence
            if np.array_equal(self.labels, previous_labels):
                break
            
            # Update centroids and weights
            self.centroids = self._calculate_centroids(data, self.labels)
            self.weights = self._update_weights(data, self.centroids, self.labels)
            
            previous_labels = np.copy(self.labels)
    
    def get_cluster_frequencies(self) -> List[List[float]]:
        """
        Get frequency distribution of cluster assignments.
        
        Returns:
            List of [cluster_id, count] pairs
        """
        if self.labels is None:
            raise ValueError("Model must be fitted first")
        
        unique, counts = np.unique(self.labels, return_counts=True)
        return np.column_stack((unique, counts)).tolist()


def demonstrate_algorithm():
    """
    Demonstration function showing how to use the weighted clustering algorithm.
    """
    print("Custom Weighted Clustering Algorithm Demo")
    print("=" * 50)
    
    # Generate sample economic data (replace with your actual data)
    np.random.seed(42)
    n_samples = 200
    n_features = 4
    
    # Simulate economic indicators: GDP growth, employment, productivity, investment
    sample_data = np.random.randn(n_samples, n_features)
    sample_data = sample_data * [0.5, 0.3, 0.4, 0.6] + [2.0, 75.0, 100.0, 0.15]
    
    # Create data matrix
    data = DataMatrix(sample_data)
    
    print(f"Data shape: {data.data.shape}")
    print(f"Features: {data.n_features}, Samples: {data.n_samples}")
    
    # Test different parameter combinations
    print("\nTesting parameter combinations:")
    print("-" * 40)
    
    for n_clusters in range(2, 5):
        for beta_param in [1.1, 1.5, 2.0]:
            try:
                # Initialize and fit clustering model
                clusterer = WeightedClustering(n_clusters=n_clusters, beta=beta_param)
                clusterer.fit(data)
                
                # Get results
                frequencies = clusterer.get_cluster_frequencies()
                print(f"Clusters: {n_clusters}, Beta: {beta_param:.1f} → {frequencies}")
                
            except Exception as e:
                print(f"Error with k={n_clusters}, beta={beta_param}: {e}")


if __name__ == "__main__":
    # Run demonstration when script is executed directly
    demonstrate_algorithm()
