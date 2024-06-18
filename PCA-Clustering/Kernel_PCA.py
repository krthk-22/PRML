import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datasets import load_dataset
import math

class Kernel_PCA():
    def __init__(self, data, kernel, kernel_parameter, num_images):
        self.train_data = data['train']
        self.test_data = data['test']
        self.kernel_type = kernel
        self.kernel_parameter = kernel_parameter
        self.eigen_values = None
        self.eigen_vectors = None
        self.max_explained = None
        self.percentage_importance = None
        self.cummulative_percentage = None
        self.alphas = None
        self.projections = None
        self.get_eigen_vectors(num_images)
        self.get_percentage_importance()

    def get_labelled_data(self, data, num_images):
        labelled_data = [[] for i in range(10)]
        for single_data_point in data:
            image = single_data_point['image']
            label = single_data_point['label']
            labelled_data[label].append(np.array(image).flatten())
        labelled_data = np.array([labelled_data[i][:num_images] for i in range(10)], dtype='float64')
        return labelled_data.reshape(-1, labelled_data.shape[-1]).T
    
    def polyKernel(self, data):
        kernelised_data = np.zeros((data.shape[-1], data.shape[-1]))
        for i in range(data.shape[-1]):
            for j in range(data.shape[-1]):
                kernelised_data[i, j] = (1 + np.dot(data[:, i], data[:, j]))**self.kernel_parameter
        kernelised_data = self.center_kernel(kernelised_data)
        return kernelised_data
    
    def radialKernel(self, data):
        kernelised_data = np.zeros((data.shape[-1], data.shape[-1]))
        for i in range(data.shape[-1]):
            for j in range(data.shape[-1]):
                diff = data[:, i] - data[:, j]
                kernelised_data[i, j] = np.exp((-(np.dot(diff, diff))/(2*(self.kernel_parameter**2))))
        kernelised_data = self.center_kernel(kernelised_data)
        return kernelised_data

    def center_kernel(self, data):
        one_n = np.ones((data.shape[-1], data.shape[-1]))/data.shape[-1]
        centered_kernelised_data = np.array(data - np.dot(one_n, data) - np.dot(data, one_n) + np.dot(np.dot(one_n, data), one_n))
        epsilon = 1e-8
        centered_kernelised_data += epsilon*np.eye(data.shape[-1])
        return centered_kernelised_data


    def kernelise(self, data):
        if self.kernel_type == "Polynomial":
            return self.polyKernel(data)
        elif self.kernel_type == "Radial":
            return self.radialKernel(data)
        
    def get_percentage_importance(self):
        eigen_sum = np.sum(self.eigen_values)
        self.percentage_importance = (self.eigen_values/eigen_sum)*100
        self.cummulative_percentage = np.cumsum(self.percentage_importance)

    def get_projections(self, data):
        self.alphas = np.array([self.eigen_vectors[:, j]/math.sqrt(max(1, self.eigen_values[j])) for j in range(data.shape[-1])])
        self.projections = np.matmul(self.alphas, data.T)

    def get_eigen_vectors(self, num_images):
        labelled_train_data = self.get_labelled_data(self.train_data, num_images)
        kernelised_data = self.kernelise(labelled_train_data)
        self.eigen_values, self.eigen_vectors = np.linalg.eig(kernelised_data)
        descending_indices = np.argsort(self.eigen_values)[::-1]
        self.eigen_values = self.eigen_values[descending_indices]
        self.eigen_vectors = self.eigen_vectors[:, descending_indices]
        self.get_projections(kernelised_data)

    def get_principal_components(self, explanation_needed = 95):
        num_components = np.sum(self.cummulative_percentage < explanation_needed)
        self.max_explained = num_components+1
        return num_components+1, self.get_first_eigenvectors(num_components+1)
    
    def get_first_eigenvectors(self, num_components):
        return self.eigen_vectors[:num_components]
    
    def show_variance_summary(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        x = np.linspace(1, 1000, 1000)
        y = np.array([self.percentage_importance, self.percentage_importance, self.cummulative_percentage])
        y_labels = [r"% variance explained by $n^{th}$ EV", r"% variance explained by $n^{th}$ EV", r"Cummulative % explained till $n^{th}$ EV"]
        annotate = [str(self.max_explained) + " features", str(self.max_explained) + " features",
                    str(round(self.cummulative_percentage[self.max_explained], 2)) + "% Explained\n"  + str(self.max_explained) + " Features"]
        xy = [(+20, +20), (+20, +20), (-20, -40)]
        for i, axis in enumerate(axes):
            axis.plot(x, y[i])
            axis.plot(self.max_explained-1, y[i][self.max_explained-1], 'r*')
            axis.annotate(annotate[i],  fontsize=10, family="serif",
                        xy=(self.max_explained-1, y[i][self.max_explained-1]), xycoords="data",
                        xytext=xy[i], textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))
            axis.set_xlabel(r"$n^{th}$ largest eigen value")
            axis.axvline(x=self.max_explained, linestyle='--', label="Important eigen values", color='red')
            axis.set_ylabel(y_labels[i])
        save_as =  self.kernel_type + ' Variance Summary ' + str(self.kernel_parameter)
        plt.savefig('images/' + save_as)

    def show_projection_correlation(self):
        fig, axes = plt.subplots(1, 3, figsize = (18, 4))
        for j in range(3):
            axes[j].scatter(self.projections[j], self.projections[(j+1)%3])
            axes[j].ticklabel_format(scilimits=(0, 4))
            axes[j].set_xlabel("Projections along the " + str(j+1) + " PC")
            axes[j].set_ylabel("Projections along the " + str((j+2)%3) + " PC")
            axes[j].set_title( self.kernel_type + " Kernel with d = " + str(self.kernel_parameter))

        save_as =  self.kernel_type + ' PC correlation ' + str(self.kernel_parameter)
        plt.savefig('images/' + save_as)



mnist_data = load_dataset('mnist')
num_images = 100
exponents = [2, 3, 4]
for exponent in exponents:
    poly_mnist_pca = Kernel_PCA(mnist_data, "Polynomial", exponent, num_images)
    max_explained, principal_components = poly_mnist_pca.get_principal_components(95)
    poly_mnist_pca.show_variance_summary()
    poly_mnist_pca.show_projection_correlation()

sigmas = [1000, 2000, 3000]
for sigma in sigmas:
    rad_mnist_pca = Kernel_PCA(mnist_data, "Radial", sigma, num_images)
    max_explained, principal_components = rad_mnist_pca.get_principal_components(95)
    rad_mnist_pca.show_variance_summary()
    rad_mnist_pca.show_projection_correlation()