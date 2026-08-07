import numpy as np
import matplotlib.pyplot as plt

noise = np.array([0, 0.01, 0.02, 0.05, 0.08, 0.1])

GSP = np.array([0.2410, 0.1510, 0.0975, 0.0366, 0.0169, 0.0136])
GSP_error = np.array([0.0587, 0.0412, 0.0243, 0.0037, 0.0012, 0.0019])

ARI = np.array([1, 1, 1, 1, 0.5959, 0.6229])
ARI_error = np.array([0,0,0,0,0.2021, 0.3362])

plt.figure(figsize=(10,10))
plt.errorbar(noise, GSP, yerr=GSP_error, fmt='o-', color='orange', capsize=3, alpha=0.7)
plt.xlabel('Noise')
plt.title('GSP Noise scan N=8, p=2')
plt.xticks(noise)
plt.ylabel('GSP')

hits = 8
baseline = 2 / (2 ** (hits))
plt.axhline(baseline, linestyle='--', label=f'Uniform baseline {baseline:.4f}')
plt.legend()
plt.show()


plt.figure(figsize=(10,10))
plt.errorbar(noise, ARI, yerr=ARI_error, fmt='-o', color='red', capsize=3, alpha=0.7)
plt.xlabel('Noise')
plt.title('ARI Noise scan N=8, p=2')
plt.xticks(noise)
plt.ylabel('ARI')
plt.show()
