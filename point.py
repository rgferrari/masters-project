import math
import numpy as np

class Point():
    points = []

    def __init__(self, features: list, label: int):
        self.original_features = features
        self.features = features
        self.label = label
        self.best_label = None
        self.weights = {}
        self.teammates = []
        self.disputed_enemies = []
        self.strength = 0
        self.dispute_result = 0


    @classmethod
    def get_points(cls):
        return cls.points
    

    @classmethod
    def set_points(cls, points):
        cls.points = points


    @classmethod
    def create_points(cls, X):
        points = [Point(features=x, label=i) for i, x in enumerate(X)]
        cls.set_points(points)

        for point in points:
            point.initialize_weights()
            point.compute_best_label()

        for point in points:
            point.update_label()
            point.update_teammates()
            point.compute_strength()

        return points


    def initialize_weights(self):
        total_potential_strength = 0

        # Calculate the inverse distances and the total inverse distance
        for point in self.points:
            if point == self:
                self.weights[point] = 0.0
            else:
                potential_strength = self.compute_strength_gain(point)
                self.weights[point] = potential_strength
                total_potential_strength += potential_strength

        # Normalize the weights to be in the range [0, 1]
        for point in self.points:
            if total_potential_strength > 0:
                self.weights[point] /= total_potential_strength


    def update_weights(self, learning_rate: float):
        for teammate in self.teammates:
            self.weights[teammate] += learning_rate * self.dispute_result

        for enemy, battle_result in self.disputed_enemies:
            self.weights[enemy] -= learning_rate * battle_result
        
        self.disputed_enemies = []
        self.dispute_result = 0


    def move_towards_team(self, step_size: float):
        """Move the point towards the average position of the teammates, excluding those beyond one standard deviation."""

        if not self.teammates:
            return

        # Calculate the average position
        avg_position = np.mean([teammate.features for teammate in self.teammates], axis=0)
        
        # Calculate the standard deviation of the positions
        std_dev = np.std([teammate.features for teammate in self.teammates], axis=0)
        
        # Filter teammates within one standard deviation of the average position
        filtered_teammates = [
            teammate for teammate in self.teammates
            if np.all(np.abs(teammate.features - avg_position) <= std_dev)
        ]
        
        if not filtered_teammates:
            return

        # Calculate the new average position with the filtered teammates
        filtered_avg_position = np.mean([teammate.features for teammate in filtered_teammates], axis=0)

        # Move the point towards the new average position
        self.features = [
            self.features[i] + step_size * (filtered_avg_position[i] - self.features[i])
            for i in range(len(self.features))
        ]


    def compute_best_label(self):
        """Rank the labels according to the weights."""

        labels_rank = {}

        for point, weight in self.weights.items():
            if point.label not in labels_rank:
                labels_rank[point.label] = 0
            labels_rank[point.label] += weight

        self.best_label = max(labels_rank, key=labels_rank.get)


    def update_label(self):
        """Update the label according to the relationships with other points."""

        self.label = self.best_label


    def compute_distance(self, other: 'Point'):
        """Compute the Euclidean distance between two points."""

        if len(self.features) != len(other.features):
            raise ValueError("Points must have the same number of features")

        return math.sqrt(sum((a - b) ** 2 
                         for a, b in zip(self.features, other.features)))


    def compute_strength_gain(self, point):
        distance = self.compute_distance(point)

        # To avoid division by zero 
        # and to prevent the strength from being too high
        if distance < 0.1:
            distance = 0.1
        
        return 1 / distance
        

    def compute_strength(self):
        """The point stregth is the sum of the distances to all other points 
        with the same label.
        """

        self.strength = 0
        for teammate in self.teammates:
            self.strength += self.compute_strength_gain(teammate)


    def update_teammates(self):
        """Update the list of teammates according to the other points labels."""
            
        self.teammates = [
            point for point in self.points 
            if point.label == self.label 
            and point != self
        ]


    def find_closest_non_teammates(self, n):
        non_teammates = [point for point in self.points 
                         if point.label != self.label]
        non_teammates.sort(key=lambda point: self.compute_distance(point))
        return non_teammates[:n]

    
    def dispute(self, n):
        """The point with the highest strength wins the dispute."""

        closest_non_teammates = self.find_closest_non_teammates(n)

        for other in closest_non_teammates:
            battle_result = self.strength - other.strength
            self.dispute_result += battle_result
            self.disputed_enemies.append((other, battle_result))