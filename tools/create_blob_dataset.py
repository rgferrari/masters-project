from sklearn.datasets import make_blobs
import pandas as pd

def create_blobs_dataset(n_samples=256, centers=2, random_state=42):
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)
    df = pd.DataFrame(X, columns=['x', 'y'])
    df['label'] = y
    df.to_csv('datasets/blobs.csv', index=False)
    print("Blobs dataset generated and saved to 'datasets/blobs.csv'")

create_blobs_dataset()