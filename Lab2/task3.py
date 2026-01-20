from data import overlapping_data
from visual import overlapping_data_plot
from visual import svm_with_soft_margin

def main():
    X, y = overlapping_data()
    overlapping_data_plot(X, y)
    C = [0.1, 1, 10, 100]
    for i in range(len(C)):
        svm_with_soft_margin(X, y, C=C[i])

if __name__ == "__main__":
    main()