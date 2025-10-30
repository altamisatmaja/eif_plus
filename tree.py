"""Isolation Tree implementation for EIF⁺."""

import numpy as np
from .utils import average_path_length, generate_random_hyperplane

class IsolationTree:
    """A single isolation tree in the EIF⁺ forest."""
    
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None
        
    def build(self, X, depth, random_state, eta):
        """
        Build the isolation tree recursively.
        
        Parameters:
        -----------
        X : ndarray
            Input data
        depth : int
            Current depth
        random_state : RandomState
            Random state instance
        eta : float
            Hyperparameter controlling hyperplane selection spread
            
        Returns:
        --------
        dict
            Tree node
        """
        n_samples, n_features = X.shape
        
        # Stop conditions
        if depth >= self.max_depth or n_samples <= 1:
            return self._create_leaf(n_samples, depth)
        
        # Select hyperplane using EIF⁺ strategy
        normal_vector, intercept_point = self._select_hyperplane_eif_plus(
            X, random_state, eta, n_features
        )
        
        # Split data
        projections = X @ normal_vector
        threshold = intercept_point @ normal_vector
        
        left_mask = projections <= threshold
        right_mask = ~left_mask
        
        # Check if split is valid
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return self._create_leaf(n_samples, depth)
        
        # Recursively build subtrees
        left_tree = self.build(X[left_mask], depth + 1, random_state, eta)
        right_tree = self.build(X[right_mask], depth + 1, random_state, eta)
        
        return self._create_node(
            normal_vector, intercept_point, threshold, 
            left_tree, right_tree, n_samples, depth
        )
    
    def _select_hyperplane_eif_plus(self, X, random_state, eta, n_features):
        """
        Select hyperplane using EIF⁺ strategy.
        """
        # Generate random normal vector
        normal_vector = generate_random_hyperplane(n_features, random_state)
        
        # Project data onto normal vector
        projections = X @ normal_vector
        
        # EIF⁺: Sample from normal distribution around mean projection
        mean_proj = np.mean(projections)
        std_proj = np.std(projections)
        
        # Sample alpha from N(mean_proj, eta * std_proj)
        alpha = random_state.normal(mean_proj, eta * std_proj)
        
        # Ensure alpha is within data range
        min_proj, max_proj = np.min(projections), np.max(projections)
        alpha = np.clip(alpha, min_proj, max_proj)
        
        # Calculate intercept point
        intercept_point = alpha * normal_vector
        
        return normal_vector, intercept_point
    
    def _create_leaf(self, size, depth):
        """Create a leaf node."""
        return {
            'type': 'leaf',
            'size': size,
            'depth': depth
        }
    
    def _create_node(self, normal_vector, intercept_point, threshold, left, right, size, depth):
        """Create an internal node."""
        return {
            'type': 'node',
            'normal_vector': normal_vector,
            'intercept_point': intercept_point,
            'threshold': threshold,
            'left': left,
            'right': right,
            'size': size,
            'depth': depth
        }
    
    def path_length(self, x, node=None):
        """
        Calculate path length for a sample in this tree.
        
        Parameters:
        -----------
        x : ndarray
            Single sample
        node : dict, optional
            Current node (for recursion)
            
        Returns:
        --------
        float
            Path length
        """
        if node is None:
            node = self.root
            
        if node['type'] == 'leaf':
            return node['depth'] + average_path_length(node['size'])
        
        # Determine which side the point falls on
        projection = x @ node['normal_vector']
        if projection <= node['threshold']:
            return self.path_length(x, node['left'])
        else:
            return self.path_length(x, node['right'])