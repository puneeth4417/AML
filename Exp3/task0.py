from data import load_breast_cancer_data
from data import load_breast_cancer_extended
from visual import plot_2d_data

def data():
    """A function to visulize the dataset"""
    X, y = load_breast_cancer_data()
    plot_2d_data(X, y, title="Breast Cancer Data")
    
def extended_data():
    X, y = load_breast_cancer_extended()      
    plot_2d_data(X, y, title="Breast Cancer Data ({n_samples} Samples, PCA Reduced)")
    
if __name__=="__main__":
    data()
    extended_data()