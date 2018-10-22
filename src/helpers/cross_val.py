"""
Cross validation helpers.
"""
import numpy as np

def build_k_indices(min_idx, n_data, k_fold):
    """
    Build k indices for k-fold.
    """
    interval = int((n_data - min_idx) / k_fold)
    indices = np.random.permutation(np.arange(min_idx, n_data + 1))
    
    # If only one fold, take 25% for validation to simulate simple splitting
    k_indices = []
    if k_fold == 1:
        pivot = int(interval / 4)
        k_indices.append((indices[:pivot], indices[pivot:]))
    else:
        for k in range(k_fold):
            low, high = k * interval, (k + 1) * interval
            val_indices = indices[low : high]
            train_indices = np.append(indices[:low], indices[high:])
            k_indices.append((val_indices, train_indices))
            
    return k_indices