import pandas as pd
import numpy as np

# Parameters
num_samples = 256  # Number of samples to generate

# Generate random data
np.random.seed(42)  # For reproducibility
x = np.random.uniform(low=0.0, high=10.0, size=num_samples)
y = np.random.uniform(low=0.0, high=30.0, size=num_samples)
attribute = np.random.randint(1, 4, size=num_samples)
label = np.random.randint(1, 3, size=num_samples)

# Create DataFrame
df = pd.DataFrame({
    'x': x,
    'y': y,
    'attribute': attribute,
    'label': label
})

# Save to CSV
df.to_csv('datasets/random_dataset.csv', index=False)

print("Dataset generated and saved to 'random_dataset.csv'")