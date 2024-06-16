from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

class PCA():
    def __init__(self, data, num_images):
        self.train_data = data['train']
        self.test_data = data['test']
        self.feature_mean = None
        self.eigen_values = None
        self.eigen_vectors = None
        self.max_explained = None
        self.percentage_importance = None
        self.cummulative_percentage = None
        self.projections = None
        self.get_eigen_vectors(num_images)
        self.get_percentage_importance()
        
    def get_labelled_data(self, data, num_images):
        labelled_data = [[] for i in range(10)]
        for single_data_point in data:
            image = single_data_point['image']
            label = single_data_point['label']
            labelled_data[label].append(np.array(image).flatten())
        labelled_data = np.array([labelled_data[i][:num_images] for i in range(10)])
        return labelled_data.reshape(-1, labelled_data.shape[-1]).T
    
    def center_data(self, data):
        self.feature_mean = np.mean(data, axis=1, keepdims=True)
        return data - self.feature_mean
    
    def get_eigen_vectors(self, num_images):
        labelled_train_data = self.get_labelled_data(self.train_data, num_images)
        centered_train_data = self.center_data(labelled_train_data)
        covariance_matrix = np.matmul(centered_train_data, centered_train_data.T, dtype='float64')
        covariance_matrix /= 10*num_images
        covariance_matrix += (1e-6)*np.eye(covariance_matrix.shape[0])

        self.eigen_values, self.eigen_vectors = np.linalg.eigh(covariance_matrix)
        descending_indices = np.argsort(self.eigen_values[::-1])
        self.eigen_values = self.eigen_values[descending_indices]
        self.eigen_vectors = self.eigen_vectors[:, descending_indices]
        self.get_projections(centered_train_data)

    def get_percentage_importance(self):
        eigen_sum = np.sum(self.eigen_values)
        self.percentage_importance = (self.eigen_values/eigen_sum)*100
        self.cummulative_percentage = np.cumsum(self.percentage_importance)

    def get_principal_components(self, explanation_needed = 95):
        num_components = np.sum(self.cummulative_percentage < explanation_needed)
        self.max_explained = num_components+1
        return num_components+1, self.get_first_eigenvectors(num_components+1)
    
    def get_first_eigenvectors(self, num_components):
        return self.eigen_vectors[:num_components]
    
    def show_principal_components(self):
        fig, axes = plt.subplots(28, 28, figsize=(28, 28))
        for i, pc in enumerate(self.eigen_vectors.T):
            q, r = i//28, i%28
            axes[q][r].imshow(pc.reshape((28, 28)), cmap='gray')
            axes[q][r].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        plt.savefig('Principal Components')
    
    def show_variance_summary(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        x = np.linspace(1, 784, 784)
        y = np.array([self.percentage_importance, self.percentage_importance, self.cummulative_percentage])
        y_labels = [r"% variance explained by $n^{th}$ EV", r"% variance explained by $n^{th}$ EV", r"Cummulative % explained till $n^{th}$ EV"]
        annotate = [str(self.max_explained) + " features", str(max_explained) + " features",
                    str(round(mnist_pca.cummulative_percentage[max_explained], 2)) + "% Explained\n"  + str(max_explained) + " Features"]
        xy = [(+20, +20), (+20, +20), (-20, -40)]
        for i, axis in enumerate(axes):
            axis.plot(x, y[i])
            axis.plot(self.max_explained-1, y[i][self.max_explained-1], 'r*')
            axis.annotate(annotate[i],  fontsize=10, family="serif",
                        xy=(self.max_explained-1, y[i][self.max_explained-1]), xycoords="data",
                        xytext=xy[i], textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))
            axis.set_xlim(-5, 785)
            axis.set_xlabel(r"$n^{th}$ largest eigen value")
            axis.axvline(x=max_explained, linestyle='--', label="Important eigen values", color='red')
            axis.set_ylabel(y_labels[i])
        plt.savefig('Variance Summary')

    
    def get_projections(self, data):
        stored_data = np.matmul(data.T, self.eigen_vectors)
        projections = []
        for i in range(10*num_images):
            projections.append(np.array([stored_data[i, j]*self.eigen_vectors[:, j] for j in range(784)]))
        self.projections = np.array(projections)
    
    def reconstruct_data(self, num_PC):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for i in range(10):
            q, r = i//5, i%5
            image = np.zeros((28, 28))
            for j in range(num_PC):
                image += self.projections[100*i, j].reshape((28, 28))
            
            image += self.feature_mean.reshape((28, 28))
            axes[q][r].imshow(image, cmap='gray')
            axes[q][r].tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)
            axes[q][r].set_xlabel("Image of " + str(i) + " , d = " + str(num_PC))
        plt.savefig('Reconstructed using ' + str(num_PC))
    
mnist_data = load_dataset('mnist')
num_images = 100
mnist_pca = PCA(mnist_data, num_images)
max_explained, prinicipal_components = mnist_pca.get_principal_components(95)
mnist_pca.show_principal_components()
mnist_pca.show_variance_summary()
reconstruct_using = [30, 80, 130]
for num_PC in reconstruct_using:
    mnist_pca.reconstruct_data(num_PC)