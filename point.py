import math

class Point():
    def __init__(self, features: list, label: int):
        self.features = features
        self.label = label
        self.relationships = {}
        self.strength = 0
        self.last_dispute_result = 0


    def update_label(self):
        """Update the label according to the relationships with other points."""

        
        pass


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
        
        for teammate in self.teammates:
            self.strength += self.compute_distance(teammate)

    
    def dispute(self, other: 'Point'):
        """The point with the highest strength wins the dispute."""

        self.last_dispute_result = self.strength - other.strength