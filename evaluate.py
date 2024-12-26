import numpy as np
from data_utils import load_fmri_data
import matplotlib.pyplot as plt

# Load real and generated data
real_data = load_fmri_data('data/fmri_data.npy')
generated_data = load_fmri_data('results/generated_fmri_data.npy')

# Perform evaluation (e.g., calculate statistical measures, visualize data)
# Example: compare the mean and standard deviation of real and generated data
real_mean = np.mean(real_data, axis=0)
real_std = np.std(real_data, axis=0)
generated_mean = np.mean(generated_data, axis=0)
generated_std = np.std(generated_data, axis=0)

print("Real Data Mean:", real_mean)
print("Real Data Std:", real_std)
print("Generated Data Mean:", generated_mean)
print("Generated Data Std:", generated_std)

# Example visualization (adjust as needed for your data)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(real_data.flatten(), bins=30)
plt.title("Real Data Histogram")
plt.subplot(1, 2, 2)
plt.hist(generated_data.flatten(), bins=30)
plt.title("Generated Data Histogram")
plt.show()
