import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint

class Clustering:
    def __init__(self, dataset):
        self.x, self.y = self.get_data(dataset)
        self.points = np.array([self.x, self.y]).T
        self.distant_point = np.array([1e9, 1e9])
        self.num_points = self.points.shape[1]
    
    def show_dataset(self):
        fig, axes = plt.subplots(1, figsize=(6, 4))
        axes.scatter(self.x, self.y)
        axes.tick_params(left=False, bottom = False, labelleft=False, labelbottom=False)
        axes.set_xlabel("Given Data in XY-plane")
        plt.savefig('images/Q2/Dataset')
    
    def get_data(self, dataset):
        return np.array(dataset.iloc[:, 0].values), np.array(dataset.iloc[:, 1].values)
    
cm_dataset = pd.read_csv('cm_dataset.csv')
cm_clustering = Clustering(cm_dataset)
cm_clustering.show_dataset()