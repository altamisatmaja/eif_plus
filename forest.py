"""Enhanced Extended Isolation Forest implementation."""

import numpy as np
from sklearn.utils import check_random_state
from .base import BaseIsolationForest
from .tree import IsolationTree
from .utils import validate_data

class EnhancedExtendedIsolationForest(BaseIsolationForest):
    """
    Enhanced Extended Isolation Forest (EIF⁺)
    
    An improved version of Extended Isolation Forest that enhances generalization
    by modifying the hyperplane selection strategy.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.
        
    max_samples : int or 'auto', default='auto'
        The number of samples to draw from X to train each base estimator.
        If 'auto', then `max_samples=min(256, n_samples)`.
        
    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function. If 'auto', the threshold is determined as
        in the original paper.
        
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
        
    eta : float, default=1.0
        Hyperparameter controlling the spread of hyperplane selection in EIF⁺.
        Higher values allow hyperplanes to be selected further from the data mean,
        potentially improving generalization to unseen anomalies.
    
    Attributes:
    -----------
    estimators_ : list of IsolationTree
        The collection of fitted sub-estimators.
        
    max_samples_ : int
        The actual number of samples used for each base estimator.
        
    n_features_in_ : int
        Number of features seen during fit.
        
    threshold_ : float
        The threshold value for outlier detection.
        
    is_fitted_ : bool
        True if the model has been fitted.
    
    Examples:
    --------
    >>> from eif_plus import EnhancedExtendedIsolationForest
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=300, centers=1, random_state=42)
    >>> eif = EnhancedExtendedIsolationForest(contamination=0.1, random_state=42)
    >>> eif.fit(X)
    >>> predictions = eif.predict(X)
    >>> scores = eif.decision_function(X)
    """
    
    def __init__(self, n_estimators=100, max_samples='auto', 
                 contamination='auto', random_state=None, eta=1.0):
        super().__init__(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state
        )
        self.eta = eta
        
    def _fit(self, X):
        """Build the forest of isolation trees."""
        n_samples, n_features = X.shape
        self.random_state_ = check_random_state(self.random_state)
        
        # Calculate max depth
        max_depth = np.ceil(np.log2(self.max_samples_))
        
        # Build trees
        self.estimators_ = []
        for i in range(self.n_estimators):
            # Bootstrap sample
            sample_indices = self.random_state_.choice(
                n_samples, self.max_samples_, replace=False
            )
            X_sample = X[sample_indices]
            
            # Build tree
            tree = IsolationTree(max_depth)
            tree.root = tree.build(
                X_sample, depth=0, 
                random_state=self.random_state_, 
                eta=self.eta
            )
            self.estimators_.append(tree)
            
        return self
    
    def _decision_function(self, X):
        """Compute anomaly scores for samples."""
        n_samples = X.shape[0]
        path_lengths = np.zeros(n_samples)
        
        for i in range(n_samples):
            tree_paths = []
            for tree in self.estimators_:
                tree_paths.append(tree.path_length(X[i]))
            path_lengths[i] = np.mean(tree_paths)
        
        # Calculate anomaly scores
        scores = 2 ** (-path_lengths / self._c)
        return scores
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'eta': self.eta
        }
        return params
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self