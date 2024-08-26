import math
import random

class Point():
    points = []

    def __init__(self, features: list, label: int):
        self.features = features
        self.label = label
        self.best_label = None
        self.weights = {}
        self.teammates = []
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
        for point in self.points:
            if point == self:
                self.weights[point] = 0.0
            else:
                self.weights[point] = random.uniform(0.0, 1.0)


    def update_weights(self, learning_rate: float):
        for teammate in self.teammates:
            self.weights[teammate] += learning_rate * self.dispute_result

        self.dispute_result = 0


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


    def compute_strength(self):
        """The point stregth is the sum of the distances to all other points 
        with the same label.
        """

        self.strength = 0
        
        for teammate in self.teammates:
            self.strength += self.compute_distance(teammate)


    def update_teammates(self):
        """Update the list of teammates according to the other points labels."""
            
        self.teammates = [
            point for point in self.points 
            if point.label == self.label 
            and point != self
        ]

    
    def dispute(self, other: 'Point'):
        """The point with the highest strength wins the dispute."""

        self.dispute_result += self.strength - other.strength