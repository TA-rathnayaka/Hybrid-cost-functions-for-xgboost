import numpy as np
from sklearn.neighbors import NearestNeighbors

def icost_fit_predict(X_train, y_train, cfb=None, cfs=None, cfp=None, n_neighbors=6):
    # Calculate imbalance ratio and identify minority class
    class_counts = np.bincount(y_train)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)
    imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
    
    # Set default costs
    cfb = cfb or imbalance_ratio * 1.1
    cfs = cfs or imbalance_ratio * 0.6
    cfp = cfp or imbalance_ratio * 0.2
    
    # Initialize weights
    sample_weights = np.ones(len(y_train))
    minority_indices = np.where(y_train == minority_class)[0]
    
    if len(minority_indices) == 0:
        return sample_weights
    
    # KNN analysis
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X_train)
    _, indices = knn.kneighbors(X_train[minority_indices])
    
    # Count majority neighbors for each minority sample
    majority_neighbors = np.sum(y_train[indices] == majority_class, axis=1)
    
    # Apply costs based on neighbor count
    sample_weights[minority_indices] = np.where(
        majority_neighbors == 0, cfp,
        np.where(majority_neighbors == 1, cfs, cfb)
    )
    
    return sample_weights