from data import generate_linear_data
from visual import svm_with_hard_margin

def main():
    X, y = generate_linear_data()
    svm_with_hard_margin(X, y)

if __name__ == "__main__":
    main()