"""Base class for isolation forest models."""

from abc import ABC, abstractmethod
import numpy as np
from .utils import validate_data, average_path_length
from .exceptions import NotFittedError

class BaseIsolationForest(ABC):
    """Base class for isolation forest implementations."""
    
    def __init__(self, n_estimators=100, max_samples='auto', 
                 contamination='auto', random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.is_fitted_ = False
        
    @abstractmethod
    def _fit(self, X):
        """Abstract method for fitting the model."""
        pass
    
    @abstractmethod
    def _decision_function(self, X):
        """Abstract method for computing anomaly scores."""
        pass
    
    def fit(self, X, y=None):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency.
            
        Returns:
        --------
        self : object
            Fitted estimator.
        """
        X = validate_data(X)
        self.n_features_in_ = X.shape[1]
        
        # Determine max_samples
        if self.max_samples == 'auto':
            self.max_samples_ = min(256, X.shape[0])
        else:
            self.max_samples_ = min(self.max_samples, X.shape[0])
        
        self._c = average_path_length(self.max_samples_)
        
        # Fit the model
        self._fit(X)
        
        # Set threshold
        scores = self._decision_function(X)
        if self.contamination == 'auto':
            self.threshold_ = np.percentile(scores, 100 * 0.1)
        else:
            self.threshold_ = np.percentile(scores, 100 * self.contamination)
            
        self.is_fitted_ = True
        return self
    
    def decision_function(self, X):
        """
        Compute the anomaly score for each sample.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        scores : ndarray of shape (n_samples,)
            The anomaly score of each sample.
        """
        if not self.is_fitted_:
            raise NotFittedError("Model must be fitted before making predictions")
        
        X = validate_data(X, self.n_features_in_)
        return self._decision_function(X)
    
    def predict(self, X):
        """
        Predict if samples are outliers.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        is_outlier : ndarray of shape (n_samples,)
            For each observation, tells whether it should be considered
            as an outlier according to the fitted model.
        """
        scores = self.decision_function(X)
        return (scores > self.threshold_).astype(int)
    
    def fit_predict(self, X, y=None):
        """
        Fit the model and predict outliers.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency.
            
        Returns:
        --------
        is_outlier : ndarray of shape (n_samples,)
            For each observation, tells whether it should be considered
            as an outlier according to the fitted model.
        """
        return self.fit(X, y).predict(X)
    
    def score_samples(self, X):
        """
        Opposite of the anomaly score defined in the original paper.
        The higher, the more normal.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        scores : ndarray of shape (n_samples,)
            The opposite of the anomaly score of each sample.
        """
        return -self.decision_function(X)