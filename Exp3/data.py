
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import resample


def load_breast_cancer_data():
    """
    Loads breast cancer dataset and converts it to 2D using PCA
    """
    data = load_breast_cancer()
    X_2d = data.data
    y = data.target   # 0 = malignant, 1 = benign

    # pca = PCA(n_components=2)
    # X_2d = pca.fit_transform(X)

    return X_2d, y

def load_breast_cancer_extended(n_samples=200000, n_features=20):
    """
    1. Increase number of data points using resampling
    2. Reduce number of features using PCA
    3. Reduce to 2D for visualization
    """

    data = load_breast_cancer()
    X = data.data
    y = data.target

    # -------- Increase data points --------
    X_big, y_big = resample(
        X, y,
        replace=True,
        n_samples=n_samples,
        random_state=42
    )

    # -------- Reduce features --------
    pca_features = PCA(n_components=n_features)
    X_reduced = pca_features.fit_transform(X_big)

    # -------- Reduce to 2D for plotting --------
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_reduced)

    return X_2d, y_big
