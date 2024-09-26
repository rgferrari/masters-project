import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from battle_clustering import BattleClustering


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
      

if __name__ == '__main__':
    # Setup
    X, y = get_dataset('datasets/flame.csv')

    hyperparameters = json.load(open('hyperparameters.json', 'r'))
    print('hyperparameters:', hyperparameters)

    battle_clustering = BattleClustering(hyperparameters)

    # Train
    states = battle_clustering.train(X)
    points = battle_clustering.points

    # Plot
    print('Plotting...')
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plot_points(X, y, 'Original Labels', axs[0])

    final_labels = [point.label for point in points]
    plot_points(X, final_labels, 'Final Result', axs[1])

    scatter = plot_points(X, 
                          [point.label for point in points], 
                          'Clustering Evolution', 
                          axs[2])

    interval = int(ANIMATION_DURATION // hyperparameters['epochs'])
    if interval < 200:
        interval = 200
    
    ani = animation.FuncAnimation(fig, 
                                  update_plot, 
                                  frames=hyperparameters['epochs'], 
                                  fargs=(states, scatter), 
                                  interval=interval, 
                                  blit=True)
    
    ani.save('clustering_evolution.gif', writer='pillow')

    plt.show()