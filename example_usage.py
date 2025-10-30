"""
Example usage of Enhanced Extended Isolation Forest (EIF⁺)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, load_iris
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from eif_plus import EnhancedExtendedIsolationForest

def example_synthetic_data():
    """Example with synthetic 2D data for visualization."""
    print("=== Synthetic Data Example ===")
    
    # Generate data with outliers
    X, y = make_blobs(n_samples=300, centers=1, cluster_std=0.5, random_state=42)
    
    # Add outliers
    rng = np.random.RandomState(42)
    X_outliers = rng.uniform(low=-6, high=6, size=(20, 2))
    X = np.vstack([X, X_outliers])
    y = np.hstack([np.zeros(300), np.ones(20)])
    
    # Fit EIF⁺
    eif = EnhancedExtendedIsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
        eta=1.0
    )
    
    predictions = eif.fit_predict(X)
    scores = eif.decision_function(X)
    
    # Metrics
    print(classification_report(y, predictions))
    print(f"ROC-AUC: {roc_auc_score(y, scores):.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', alpha=0.6, label='Inliers')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', alpha=0.6, label='Outliers')
    plt.title('True Labels')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.scatter(X[predictions == 0, 0], X[predictions == 0, 1], 
                c='blue', alpha=0.6, label='Predicted Inliers')
    plt.scatter(X[predictions == 1, 0], X[predictions == 1, 1], 
                c='red', alpha=0.6, label='Predicted Outliers')
    plt.title('EIF⁺ Predictions')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Anomaly Score')
    plt.title('Anomaly Scores')
    
    plt.tight_layout()
    plt.show()

def example_real_data():
    """Example with real dataset (Iris)."""
    print("\n=== Real Data Example (Iris) ===")
    
    # Load and prepare data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Make it a binary anomaly detection problem
    # Consider one class as normal, others as anomalies
    normal_class = 0
    y_binary = (y != normal_class).astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit EIF⁺ with different eta values
    eta_values = [0.5, 1.0, 2.0]
    
    plt.figure(figsize=(15, 5))
    
    for i, eta in enumerate(eta_values):
        eif = EnhancedExtendedIsolationForest(
            n_estimators=100,
            contamination=0.33,  # Approximately 2/3 of classes are "anomalies"
            random_state=42,
            eta=eta
        )
        
        scores = eif.fit_predict(X_scaled)
        
        plt.subplot(1, 3, i + 1)
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=scores, cmap='coolwarm', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f'EIF⁺ Predictions (η={eta})')
        plt.xlabel('Feature 1 (standardized)')
        plt.ylabel('Feature 2 (standardized)')
    
    plt.tight_layout()
    plt.show()
    
    # Compare performance for different eta
    print("\nPerformance comparison for different η values:")
    for eta in eta_values:
        eif = EnhancedExtendedIsolationForest(
            n_estimators=100,
            contamination=0.33,
            random_state=42,
            eta=eta
        )
        scores = eif.decision_function(X_scaled)
        auc = roc_auc_score(y_binary, -scores)  # Negative because lower score = more anomalous
        print(f"η={eta}: ROC-AUC = {auc:.3f}")

def example_parameter_study():
    """Study the effect of the eta parameter."""
    print("\n=== Parameter Study ===")
    
    # Generate complex data
    X1, _ = make_blobs(n_samples=200, centers=1, cluster_std=0.8, random_state=1)
    X2, _ = make_moons(n_samples=200, noise=0.1, random_state=1)
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(200), np.ones(200)])  # Second distribution as "anomalies"
    
    # Test different eta values
    eta_range = np.linspace(0.1, 3.0, 10)
    auc_scores = []
    
    for eta in eta_range:
        eif = EnhancedExtendedIsolationForest(
            n_estimators=50,
            contamination=0.5,
            random_state=42,
            eta=eta
        )
        scores = eif.decision_function(X)
        auc = roc_auc_score(y, -scores)
        auc_scores.append(auc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(eta_range, auc_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('η (EIF⁺ hyperparameter)')
    plt.ylabel('ROC-AUC Score')
    plt.title('Effect of η Parameter on EIF⁺ Performance')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    best_eta = eta_range[np.argmax(auc_scores)]
    print(f"Best η value: {best_eta:.2f} with ROC-AUC: {np.max(auc_scores):.3f}")

if __name__ == "__main__":
    example_synthetic_data()
    example_real_data() 
    example_parameter_study()