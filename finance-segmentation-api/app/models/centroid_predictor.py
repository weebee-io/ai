import numpy as np

class CentroidPredictor:
    def __init__(self, centroids, weights, minkowski_p, sharpness_factor=2.0):
        self.centroids = centroids  # np.ndarray (shape=(num_centroids, num_features))
        self.weights = weights      # np.ndarray (shape=(num_features,))
        self.minkowski_p = minkowski_p
        self.sharpness_factor = sharpness_factor # Controls the sharpness of the probability distribution
        self.num_centroids = centroids.shape[0]
        self.num_features = centroids.shape[1]

    def _weighted_minkowski_distance(self, x_point, centroid):
        """Calculates weighted Minkowski distance between a single point and a single centroid."""
        if len(x_point) != self.num_features or len(centroid) != self.num_features:
            raise ValueError(f"Input point and centroid must have {self.num_features} features.")
        
        # Ensure weights are correctly broadcasted if x_point and centroid are 1D for a single feature comparison in a loop (not the case here but good for general functions)
        # For sum(w_i * |x_i - y_i|^p), weights should be 1D array of length num_features
        weighted_diff_p = np.sum(self.weights * (np.abs(x_point - centroid) ** self.minkowski_p))
        distance = weighted_diff_p ** (1 / self.minkowski_p)
        return distance

    def predict(self, X_scaled):
        """
        Predicts the closest centroid for each data point in X_scaled.
        X_scaled: np.ndarray (shape=(num_samples, num_features)), pre-scaled data.
        Returns: np.ndarray (shape=(num_samples,)), array of predicted centroid indices.
        """
        if X_scaled.ndim == 1: # Single sample
            X_scaled = X_scaled.reshape(1, -1)
        
        if X_scaled.shape[1] != self.num_features:
            raise ValueError(f"Input data X_scaled must have {self.num_features} features, but got {X_scaled.shape[1]}.")

        predictions = []
        for x_point in X_scaled:
            distances = [self._weighted_minkowski_distance(x_point, c) for c in self.centroids]
            predictions.append(np.argmin(distances))
        return np.array(predictions)

    def predict_proba(self, X_scaled):
        """
        Predicts class probabilities for each data point in X_scaled.
        Probabilities are derived from inverse distances (closer = higher probability).
        X_scaled: np.ndarray (shape=(num_samples, num_features)), pre-scaled data.
        Returns: np.ndarray (shape=(num_samples, num_centroids)), array of probabilities for each class.
        """
        if X_scaled.ndim == 1: # Single sample
            X_scaled = X_scaled.reshape(1, -1)

        if X_scaled.shape[1] != self.num_features:
            raise ValueError(f"Input data X_scaled must have {self.num_features} features, but got {X_scaled.shape[1]}.")

        all_probabilities = []
        for x_point in X_scaled:
            distances = np.array([self._weighted_minkowski_distance(x_point, c) for c in self.centroids])
            
            # Handle potential division by zero if a distance is exactly 0
            # Add a small epsilon to distances to prevent division by zero
            epsilon = 1e-9 
            inverse_distances = 1.0 / (distances + epsilon)

            # Apply sharpness factor to the inverse distances
            sharpened_inv_distances = inverse_distances ** self.sharpness_factor
            
            # Normalization using sharpened inverse distances
            sum_sharpened_inv_distances = np.sum(sharpened_inv_distances)
            if sum_sharpened_inv_distances == 0: # Handles cases where all sharpened inv_distances are zero
                # This might happen if distances are very large and sharpness_factor is high,
                # or if inverse_distances were somehow zero.
                # Fallback to uniform probabilities or handle as an error/warning.
                probabilities = np.ones(self.num_centroids) / self.num_centroids # Equal probability
            else:
                probabilities = sharpened_inv_distances / sum_sharpened_inv_distances
                
            all_probabilities.append(probabilities)
        return np.array(all_probabilities)

    @property
    def classes_(self):
        """Returns the class labels (centroid indices). Mimics sklearn attribute."""
        return np.arange(self.num_centroids)
