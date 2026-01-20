import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np

def plot_2d_data(X, y, title="Linearly Separable Data"):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title(title)
    plt.show()

def svm_with_hard_margin(X, y):
    svm_hard = SVC(kernel='linear', C=1e6)
    svm_hard.fit(X, y)
    print("Number of support vectors:", len(svm_hard.support_))
    w = svm_hard.coef_[0]  # weight vector
    b = svm_hard.intercept_[0]

    print("w:", w)
    print("b:", b)
    
    plt.figure(figsize=(7, 6))

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    # Highlight support vectors
    plt.scatter(
        svm_hard.support_vectors_[:, 0],
        svm_hard.support_vectors_[:, 1],
        s=120,
        facecolors='none',
        edgecolors='k', linewidth = 2,
        label='Support Vectors'
    )

    # Create x values for line plotting
    x_vals = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200)

    # Decision boundary: w.x + b = 0
    y_decision = -(w[0] * x_vals + b) / w[1]

    # Decision boundary: w.x + b = +-1
    y_margin_pos = -(w[0] * x_vals + b - 1) / w[1]
    y_margin_neg = -(w[0] * x_vals + b + 1) / w[1]

    # Plot lines
    plt.plot(x_vals, y_decision, 'k-', label='Decision Boundary')
    plt.plot(x_vals, y_margin_pos, 'k--', label='Margin +1')
    plt.plot(x_vals, y_margin_neg, 'k--', label='Margin -1')

    # Shade margin area
    plt.fill_between(x_vals, y_margin_pos, y_margin_neg, 
                    color='gray', alpha=0.2, label='Margin Area')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Hard Margin SVM with Margin Visualization')
    plt.show()
    
def overlapping_data_plot(X, y):
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
    plt.title('Overlapping Data')
    plt.show()
    
def svm_with_soft_margin(X, y, C):
    svm_soft = SVC(kernel='linear', C=C)
    svm_soft.fit(X, y)
    print("Number of support vectors:", len(svm_soft.support_))
    w = svm_soft.coef_[0]  # weight vector
    b = svm_soft.intercept_[0]

    print("w:", w)
    print("b:", b)
    
    plt.figure(figsize=(7, 6))

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    # Highlight support vectors
    plt.scatter(
        svm_soft.support_vectors_[:, 0],
        svm_soft.support_vectors_[:, 1],
        s=120,
        facecolors='none',
        edgecolors='k', linewidth = 2,
        label='Support Vectors'
    )

    # Create x values for line plotting
    x_vals = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200)

    # Decision boundary: w.x + b = 0
    y_decision = -(w[0] * x_vals + b) / w[1]

    # Decision boundary: w.x + b = +-1
    y_margin_pos = -(w[0] * x_vals + b - 1) / w[1]
    y_margin_neg = -(w[0] * x_vals + b + 1) / w[1]

    # Plot lines
    plt.plot(x_vals, y_decision, 'k-', label='Decision Boundary')
    plt.plot(x_vals, y_margin_pos, 'k--', label='Margin +1')
    plt.plot(x_vals, y_margin_neg, 'k--', label='Margin -1')

    # Shade margin area
    plt.fill_between(x_vals, y_margin_pos, y_margin_neg, 
                    color='gray', alpha=0.2, label='Margin Area')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Soft Margin SVM with Margin Visualization (C={})'.format(C))
    plt.show()