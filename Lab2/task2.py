from data import overlapping_data
from visual import overlapping_data_plot
from visual import svm_with_hard_margin

def main():
    X, y = overlapping_data()
    overlapping_data_plot(X, y)
    svm_with_hard_margin(X, y)
    
if __name__ == "__main__":
    main()