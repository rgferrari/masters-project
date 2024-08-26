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
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from point import Point

# Hyperparameters
EPOCHS = 100
LR = 1e-1
SAMPLE_PERCENTAGE = 0.1
N_INTERACTIONS = 10
ANIMATION_DURATION = 30000 # ms


def plot_points(X, y, title, ax):
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    return scatter


def update_plot(frame, X, states, scatter):
    scatter.set_array(states[frame])
    return scatter,


def get_dataset(path, header=None, sep='\t'):
    df = pd.read_csv(path, header=header, sep=sep)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y


def fit(points):
    for point in points:
        for _ in range(N_INTERACTIONS):
            other = random.choice(points)
            if other.label == point.label:
                continue

            point.dispute(other)
    
    for point in points:
        point.update_weights(LR)
        point.compute_best_label()

    for point in points:
        point.update_label()
        point.update_teammates()
        point.compute_strength()


def train(points):
    sample_size = int(len(points) * SAMPLE_PERCENTAGE)
    states = [[point.label for point in points]]
    for _ in tqdm(range(EPOCHS), desc="Training Progress"):
        sample = random.sample(points, sample_size)
        fit(sample)
        states.append([point.label for point in points])
    return states
      

if __name__ == '__main__':
    # Setup
    X, y = get_dataset('datasets/flame.csv')
    points = Point.create_points(X)

    print([point.label for point in points])

    # Train
    states = train(points)

    # Plot
    print('Plotting...')
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_points(X, y, 'Original Labels', axs[0])

    scatter = plot_points(X, [point.label for point in points], 'Clustering Evolution', axs[1])

    interval = int(ANIMATION_DURATION // EPOCHS)
    if interval < 200:
        interval = 200

    ani = animation.FuncAnimation(fig, update_plot, frames=EPOCHS, fargs=(X, states, scatter), interval=interval, blit=True)
    ani.save('clustering_evolution.gif', writer='imagemagick')
    plt.show()