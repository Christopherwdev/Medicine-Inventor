import numpy as np
import os

def load_fmri_data(filepath):
    """Loads fMRI data from a NumPy file."""
    return np.load(filepath)

def save_generated_data(data, filepath):
    """Saves generated fMRI data to a NumPy file."""
    np.save(filepath, data)

def preprocess_data(data):
    """Preprocesses the fMRI data (e.g., normalization, dimensionality reduction)."""
    # Add your preprocessing steps here (e.g., standardization, PCA)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std
    return data
