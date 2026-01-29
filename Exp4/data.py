import numpy as np

def generate_coin_toss_data(n=10, true_p=0.7, seed=1):
    np.random.seed(seed)
    data = np.random.binomial(1, true_p, n)
    
    print("Observed coin tosses (1=heads, 0=tails):\n", data)
    print("Number of heads observed:", data.sum())
    
    return data
