"""Custom exceptions for the EIF⁺ package."""

class EIFPlusError(Exception):
    """Base exception for EIF⁺ package."""
    pass

class NotFittedError(EIFPlusError):
    """Exception raised when model is used before fitting."""
    pass

class DataValidationError(EIFPlusError):
    """Exception raised for invalid data."""
    pass