
from data import load_breast_cancer_data
from data import load_breast_cancer_extended
from visual import plot_2d_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def original_data():
    """A function to train and evaluate SVM models on breast cancer data"""
    X, y = load_breast_cancer_data()

    linear_svm = SVC(kernel="linear")
    linear_svm.fit(X, y)
    y_pred_linear = linear_svm.predict(X)

    plot_2d_data(X, y_pred_linear, title="Linear SVM Prediction")
    print("Linear SVM Accuracy:", accuracy_score(y, y_pred_linear))


    poly_svm = SVC(kernel="poly", degree=2)
    poly_svm.fit(X, y)
    y_pred_poly = poly_svm.predict(X)

    plot_2d_data(X, y_pred_poly, title="Polynomial Kernel SVM (degree=2)")
    print("Polynomial Kernel SVM Accuracy:", accuracy_score(y, y_pred_poly))


    rbf_svm = SVC(kernel="rbf", gamma="scale")
    rbf_svm.fit(X, y)
    y_pred_rbf = rbf_svm.predict(X)

    plot_2d_data(X, y_pred_rbf, title="RBF Kernel SVM")
    print("RBF Kernel SVM Accuracy:", accuracy_score(y, y_pred_rbf))
    
def extended_data():
    """A function to train and evaluate SVM models on extended breast cancer data"""
    X, y = load_breast_cancer_extended()

    linear_svm = SVC(kernel="linear")
    linear_svm.fit(X, y)
    y_pred_linear = linear_svm.predict(X)

    plot_2d_data(X, y_pred_linear, title="Linear SVM Prediction")
    print("Linear SVM Accuracy:", accuracy_score(y, y_pred_linear))


    poly_svm = SVC(kernel="poly", degree=2)
    poly_svm.fit(X, y)
    y_pred_poly = poly_svm.predict(X)

    plot_2d_data(X, y_pred_poly, title="Polynomial Kernel SVM (degree=2)")
    print("Polynomial Kernel SVM Accuracy:", accuracy_score(y, y_pred_poly))


    rbf_svm = SVC(kernel="rbf", gamma="scale")
    rbf_svm.fit(X, y)
    y_pred_rbf = rbf_svm.predict(X)

    plot_2d_data(X, y_pred_rbf, title="RBF Kernel SVM")
    print("RBF Kernel SVM Accuracy:", accuracy_score(y, y_pred_rbf))
    
if __name__=="__main__":
    # original_data()
    extended_data()