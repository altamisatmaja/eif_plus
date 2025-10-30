from setuptools import setup, find_packages

setup(
    name="eif_plus",
    version="1.0.0",
    description="Enhanced Extended Isolation Forest for anomaly detection",
    author="Altamis Atmaja",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0"
    ],
    python_requires=">=3.7",
)