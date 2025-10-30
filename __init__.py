"""
Enhanced Extended Isolation Forest (EIF⁺)
A flexible and modular implementation of the Enhanced Extended Isolation Forest algorithm
for anomaly detection with improved generalization capabilities.
"""

from .forest import EnhancedExtendedIsolationForest
from .base import BaseIsolationForest
from .exceptions import EIFPlusError, NotFittedError

__version__ = "1.0.0"
__author__ = "Altamis Atmaja"
__all__ = [
    'EnhancedExtendedIsolationForest',
    'BaseIsolationForest', 
    'EIFPlusError',
    'NotFittedError'
]