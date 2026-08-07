import numpy as np
import matplotlib.pyplot as plt

noise = np.array([0, 0.01, 0.05, 0.1])

p1 = np.array([0.1337, 0.1092, 0.0507, 0.0233])
p1_error = np.array([0.0037, 0.0029, 0.0025, 0.0028])

p2 = np.array([0.2410, 0.1510, 0.0366, 0.0136])
p2_error = np.array([0.0587, 0.0412, 0.0037, 0.0019])

p3 = np.array([0.3813, 0.1686, 0.0189, 0.0099])
p3_error = np.array([0.0863, 0.0319, 0.0032, 0.0013])

plt.figure(figsize=(10,10))
plt.errorbar(noise, p1, yerr = p1_error, color='orange', label='p=1', capsize=3, alpha=0.7)

plt.errorbar(noise, p2, yerr = p2_error, color='red', label='p=2', capsize=3, alpha=0.7)

plt.errorbar(noise, p3, yerr = p3_error, color='green', label='p=3', capsize=3, alpha=0.7)

hits = 8
baseline = 2 / (2 ** (hits))
plt.axhline(baseline, linestyle='--', label=f'Uniform baseline {baseline:.4f}')

plt.legend()
plt.title('GSP dependence on noise')
plt.xlabel('Noise strength')
plt.ylabel('GSP')
plt.xticks(noise)
plt.show()
