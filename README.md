# Enhanced Extended Isolation Forest (EIF⁺)

A modular and flexible implementation of the Enhanced Extended Isolation Forest algorithm for anomaly detection with improved generalization and interpretability.

## Reference

This implementation is based on the research paper:

**Arcudi, A., Frizzo, D., Masiero, C., & Susto, G. A. (2024).**  
*Enhancing interpretability and generalizability in extended isolation forests.*  
University of Padova, Italy; Statwolf Data Science Srl, Italy.  
Available at: [https://alessioarcudi.github.io/ExIFFI/tutorial/#define-eif-model](https://alessioarcudi.github.io/ExIFFI/tutorial/#define-eif-model)

---

## Overview

The Enhanced Extended Isolation Forest (EIF⁺) builds upon the Extended Isolation Forest (EIF) framework by introducing a refined hyperplane selection mechanism that significantly improves anomaly detection performance and generalization. 

EIF⁺ modifies the random split process by drawing the hyperplane intercept scalar from a normal distribution informed by the data distribution. This allows the model to better adapt to complex data structures and identify unseen anomalies.

---

## Features

- Enhanced generalization through data-driven hyperplane selection  
- Scikit-learn compatible API (`fit()`, `predict()`, `decision_function()`)  
- Configurable `eta` parameter controlling hyperplane dispersion  
- Modular and extensible codebase  
- Lightweight and easily integrable into existing pipelines  
- Fully unsupervised anomaly detection method  

---

## Installation

### From GitHub

```bash
pip install git+https://github.com/altamisatmaja/eif_plus.git
```

## Manual Installation

```bash
git clone https://github.com/altamisatmaja/eif_plus.git
cd eif_plus
pip install -e .
```

## Usage Example
```bash
from eif_plus import EIFPlus
import numpy as np

rng = np.random.RandomState(42)
X_inliers = rng.normal(0, 1, size=(200, 2))
X_outliers = rng.uniform(-6, 6, size=(20, 2))
X = np.vstack([X_inliers, X_outliers])

model = EIFPlus(n_estimators=100, max_samples=64, eta=1.0, random_state=42)
model.fit(X)

scores = model.score_samples(X)
preds = model.predict(X, threshold=0.6)
```

## Visualization Example

```bash
import matplotlib.pyplot as plt

x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = model.score_samples(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, cmap="magma", levels=50)
plt.scatter(X_inliers[:, 0], X_inliers[:, 1], c='cyan', edgecolor='k', label='Normal')
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', edgecolor='k', label='Anomaly')
plt.colorbar(label="Anomaly Score")
plt.title("Enhanced Extended Isolation Forest (EIF⁺)")
plt.legend()
plt.show()
```

## Algorithm Description
EIF⁺ enhances the Extended Isolation Forest by modifying the hyperplane intercept sampling strategy.
```bash
α∼N(E[A],η⋅σ(A))whereA={x⋅v∣x∈X}
```
- v: random unit vector defining the hyperplane orientation
- α: intercept scalar drawn from a Gaussian distribution
- η: tunable dispersion parameter controlling generalization strength

This approach improves the adaptability of hyperplanes to the underlying data geometry, resulting in more effective isolation of anomalous instances and better generalization on unseen data.

## Citation
Arcudi, A., Frizzo, D., Masiero, C., & Susto, G. A. (2024).
Enhancing interpretability and generalizability in extended isolation forests.
University of Padova, Italy; Statwolf Data Science Srl, Italy.

## Related Resources

- Official EIF⁺ Tutorial: https://alessioarcudi.github.io/ExIFFI/tutorial/#define-eif-model
- ExIFFI Framework (Explainable EIF⁺): https://github.com/alessioarcudi/ExIFFI
- Original Isolation Forest: Liu, T., Ting, K. M., & Zhou, Z. (2008). Isolation Forest.