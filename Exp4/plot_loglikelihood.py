import numpy as np
import matplotlib.pyplot as plt
from data import generate_coin_toss_data
from likelihood import log_likelihood

data = generate_coin_toss_data(n=10, true_p=0.7, seed=1)

p_values = np.linspace(0.01, 0.99, 100)
ll_values = [log_likelihood(p, data) for p in p_values]

plt.plot(p_values, ll_values)
plt.xlabel('Probability of Heads (p)')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood vs Probability of Heads')

p_mle = data.mean()
plt.axvline(p_mle, linestyle='--')
print("MLE estimate of probability of heads:", p_mle)

plt.show()
