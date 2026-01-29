import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

true_p = 0.7
sample_sizes = [10, 30, 50, 100, 500, 1000]
p_mle_values = []

for n in sample_sizes:
    data = np.random.binomial(1, true_p, n)
    p_mle = data.mean()
    p_mle_values.append(p_mle)

plt.plot(sample_sizes, p_mle_values, marker='o')
plt.axhline(true_p, linestyle='--', label='True p')
plt.xlabel('Sample Size')
plt.ylabel('MLE Estimate of p')
plt.title('MLE convergence with increasing Sample Size')
plt.legend()
plt.show()
