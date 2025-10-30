"""Utility functions for EIF⁺."""

import numpy as np
from sklearn.utils import check_random_state
from .exceptions import DataValidationError

def validate_data(X, expected_n_features=None):
    """
    Validate and convert input data to numpy array.
    
    Parameters:
    -----------
    X : array-like
        Input data
    expected_n_features : int, optional
        Expected number of features if model is already fitted
        
    Returns:
    --------
    X_array : ndarray
        Validated numpy array
    """
    if hasattr(X, 'values'):
        X = X.values
    X_array = np.asarray(X)
    
    if X_array.ndim != 2:
        raise DataValidationError(f"Expected 2D array, got {X_array.ndim}D array")
    
    if expected_n_features is not None and X_array.shape[1] != expected_n_features:
        raise DataValidationError(
            f"Expected {expected_n_features} features, got {X_array.shape[1]}"
        )
    
    return X_array

def average_path_length(n):
    """
    Calculate the average path length for a given sample size.
    
    Parameters:
    -----------
    n : int
        Sample size
        
    Returns:
    --------
    float
        Average path length
    """
    if n <= 1:
        return 0
    elif n == 2:
        return 1
    else:
        return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1)) / n

def generate_random_hyperplane(n_features, random_state):
    """
    Generate a random hyperplane for splitting.
    
    Parameters:
    -----------
    n_features : int
        Number of features
    random_state : RandomState
        Random state instance
        
    Returns:
    --------
    normal_vector : ndarray
        Normal vector of the hyperplane
    """
    normal_vector = random_state.normal(0, 1, n_features)
    norm = np.linalg.norm(normal_vector)
    if norm == 0:
        # Fallback to axis-aligned split if normal vector is zero
        normal_vector = np.zeros(n_features)
        normal_vector[random_state.randint(0, n_features)] = 1.0
    else:
        normal_vector = normal_vector / norm
    return normal_vector