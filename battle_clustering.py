import random
from tqdm import tqdm
from typing import List

from point import Point

class BattleClustering:
    def __init__(self, kwargs) -> None:
        self.epochs = kwargs['epochs']
        self.lr = kwargs['lr']
        self.step = kwargs['step']
        self.n_disputes = kwargs['n_disputes']
        self.sample_size = kwargs['sample_size']
        self.update_teams_freq = kwargs['update_teams_freq']


    def fit(self, points: List[Point], is_updating_teams: bool=False) -> None:
        for point in points:
            point.dispute(self.n_disputes)
        
        for point in points:
            point.update_weights(self.lr)
            point.compute_best_label()

        if is_updating_teams:
            for point in points:
                point.update_label()

            for point in points:
                point.update_teammates()

        for point in points:
            point.move_towards_team(self.step)

        for point in points:
            point.compute_strength()


    def train(self, X: list) -> List[List[tuple]]:
        self.points = Point.create_points(X)

        sample_size = int(len(self.points) * self.sample_size)
        states = [[(point.features[0], point.features[1], point.label) 
                   for point in self.points]]
        
        for i in tqdm(range(self.epochs), desc="Training Progress"):
            sample = random.sample(self.points, sample_size)
            if i % self.update_teams_freq == 0:
                self.fit(sample, is_updating_teams=True)
            else:
                self.fit(sample)
            states.append([(point.features[0], point.features[1], point.label) 
                           for point in self.points])
        return states