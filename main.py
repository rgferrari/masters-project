# The idea is to create a clustering algorithm based in relationships
# Each point goal is to gatter alies to survive a combat
# When a point survive a combat, it strenghtens its relationships with its
# teammates.
# If a point loses a combat, it will weaken its relationships with its 
# teammates.
# The strength of a point is based in the distance from its teammates, thus, the
# closer the teammates, the stronger the point.
#
# 1. Initialize the points with unique labels and random relationships between 
# them
# 2. Update the labels according to the relationships
# 3. The point will get the label with strongest relationship
# 4. It can exist a table with the total relationship with that label
# 5. At the first iteration, the point will get the label of the point with the 
# highest relationship, because of the unique labels
# 6. The train begins
# 7. The points will dispute with the closest n points with different labels
# 8. The point wins if its strength is higher than the other point
# 9. The point's strength is the sum of the distances to all other points with
# the same label (this will encourage points to team up with closer ponints)
# 10. After the combat, the point will store a score with the sum of all disputes
# 11. The dispute result is the point's strength minus the other point's 
# strength, thus, if it loses, the result will be negative
# 12. The point will update its relationships with its teammates. The update
# will be proportional to the dispute result (e.g.: update * dispute_result)
# 13. After updating its relationships, the point will update its label
# 14. Go back to step 7

import random
import pandas as pd
import matplotlib.pyplot as plt

from point import Point

# Hyperparameters
# N_POINTS = 1000
# N_FEATURES = 2
# N_CLUSTERS = 3

EPOCHS = 100
SAMPLE_PERCENTAGE = 0.1
N_INTERACTIONS = 10
# SAMPLE_SIZE = int(N_POINTS * SAMPLE_PERCENTAGE)


def plot_points(X, y, title, ax):
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    return scatter


def get_dataset(path, header=None, sep='\t'):
    df = pd.read_csv(path, header=header, sep=sep)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y


def instantiate_points(X):
    return [Point(features=x, label=i) for i, x in enumerate(X)]


# relationships are annalog to weights
def initialize_relationships(points):
    for point in points:
        for other in points:
            point.relationships[other] = random.uniform(0.0, 1.0)

        point.update_label()


# this is annalog to the predict method
def combat(points):
    pass


def train(points):
    sample_size = int(len(points) * SAMPLE_PERCENTAGE)
    for epoch in range(EPOCHS):
        sample = random.sample(points, sample_size)
        combat(points)
        


if __name__ == '__main__':
    X, y = get_dataset('datasets/flame.csv')
    points = instantiate_points(X)
    initialize_relationships(points)

    train(points)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_points(X, y, 'Original Labels', axs[0])
    predicted_labels = [point.label for point in points]
    plot_points(X, predicted_labels, 'Predicted Labels', axs[1])
    plt.show()
    