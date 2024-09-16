import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from point import Point

# Hyperparameters
EPOCHS = 400
LR = 1e-2
STEP_SIZE = 1e-2
SAMPLE_PERCENTAGE = 0.1
N_INTERACTIONS = 5

ANIMATION_DURATION = 30000 # ms


def plot_points(X, y, title, ax):
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    return scatter


def update_plot(frame, states, scatter):
    # Extract the x and y positions and labels for the current frame
    xy = [(x, y) for x, y, _ in states[frame]]
    labels = [label for _, _, label in states[frame]]
    
    # Update the scatter plot
    scatter.set_offsets(xy)
    scatter.set_array(labels)
    return scatter,


def get_dataset(path):
    df = pd.read_csv(path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y


def fit(points):
    for point in points:
        point.dispute(N_INTERACTIONS)
    
    for point in points:
        point.update_weights(LR)
        point.compute_best_label()

    for point in points:
        point.update_label()
        point.update_teammates()
        point.move_towards_team(STEP_SIZE)
        point.compute_strength()


def train(points):
    sample_size = int(len(points) * SAMPLE_PERCENTAGE)
    states = [[(point.features[0], point.features[1], point.label) for point in points]]
    for _ in tqdm(range(EPOCHS), desc="Training Progress"):
        sample = random.sample(points, sample_size)
        fit(sample)
        states.append([(point.features[0], point.features[1], point.label) for point in points])
        #print([point.label for point in points])
    return states
      

if __name__ == '__main__':
    # Setup
    X, y = get_dataset('datasets/blobs.csv')
    points = Point.create_points(X)

    print([point.label for point in points])

    # Train
    states = train(points)

    # Plot
    print('Plotting...')
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plot_points(X, y, 'Original Labels', axs[0])

    final_labels = [point.label for point in points]
    plot_points(X, final_labels, 'Final Result', axs[1])

    scatter = plot_points(X, [point.label for point in points], 'Clustering Evolution', axs[2])

    interval = int(ANIMATION_DURATION // EPOCHS)
    if interval < 200:
        interval = 200
    
    ani = animation.FuncAnimation(fig, update_plot, frames=EPOCHS, fargs=(states, scatter), interval=interval, blit=True)
    ani.save('clustering_evolution.gif', writer='pillow')

    plt.show()