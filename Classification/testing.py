import os
import re
import csv
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

class Naive_Bayes():
    def __init__(self, dimensions):
        self.prior = np.zeros(2)
        self.probab = np.ones((2, dimensions), dtype='float64')
        self.comparator_vec = np.zeros(dimensions, dtype='float64')
        self.comparator_const = float(0)

    def update_parameters(self, data, labels):
        nums = np.array([labels.shape[0] - np.sum(labels)+2, np.sum(labels)+2])
        for i in range(2):
            self.prior[i] = nums[i]/np.sum(nums)
            self.probab[i] += np.sum(data[labels == i], axis = 0)
        
        for i, num in enumerate(nums):
            self.probab[i] /= num
        self.update_comparator()
    
    def update_comparator(self):
        self.comparator_const += np.log(self.prior[1]/self.prior[0])
        self.comparator_vec = np.log((self.probab[1]*(1 - self.probab[0]))/(self.probab[0]*(1 - self.probab[1])), dtype='float64')
        self.comparator_const += np.sum(np.log((1 - self.probab[1])/(1 - self.probab[0]), dtype='float64'))

    def compare_probab(self, point):
        return (np.dot(self.comparator_vec, point)+self.comparator_const > 0)

    def predict_labels(self, data):
        labels = np.zeros(data.shape[0], dtype='int')
        for i, point in enumerate(data):
            labels[i] = 1*(self.compare_probab(point))
        return labels

class Gaussian_Naive_Bayes():
    def __init__(self, dimensions):
        self.prior = np.zeros(2)
        self.means = np.zeros((2, dimensions), dtype='float64')
        self.covariance = np.zeros((dimensions, dimensions), dtype='float64')
        self.cov_inverse = np.zeros((dimensions, dimensions), dtype='float64')
        self.comparator_vec = np.zeros(dimensions)
        self.comparator_const = float(0)

    def update_parameters(self, data, labels):
        nums = np.array([labels.shape[0] - np.sum(labels), np.sum(labels)])
        for i, num in enumerate(nums):
            self.means[i] = np.mean(data[labels == i], axis=0)
            self.prior[i] = num/np.sum(nums)
        shifted_data = np.array([point - self.means[labels[i]] for i, point in enumerate(data)])
        self.covariance = np.matmul(shifted_data.T, shifted_data)
        self.covariance /= labels.shape[0]
        self.cov_inverse = np.linalg.inv(self.covariance+1e-6*np.eye(data.shape[-1]))
        
    def update_comparator(self):
        self.comparator_vec = 2*np.matmul((self.means[1] - self.means[0]).T, self.cov_inverse)
        self.comparator_const = np.log(self.prior[0]/self.prior[1]) + np.matmul(self.means[0].T, np.matmul(self.cov_inverse, self.means[0])) 
        - np.matmul(self.means[1].T, np.matmul(self.cov_inverse, self.means[1]))

    def compare_probab(self, point):
        return (np.dot(self.comparator_vec, point) + self.comparator_const >= 0)

    def predict_labels(self, data):
        labels = np.zeros(data.shape[0], dtype='int')
        for i, point in enumerate(data):
            labels[i] = 1*(self.compare_probab(point))
        return labels

class Perceptron():
    def __init__(self, w, breaking):
        '''If breaking is set true the perceptron breaks after every error and again loops from the first data point
            If breaking is set false the perceptron continues updating all the datapoints.'''
        self.w = w
        self.breaking = breaking

    def update_perceptron(self, x, y):
        self.w += x*(2*y - 1)

    def progess_perceptron(self, data, labels, max_iterations):
        iterations  = 0
        while iterations < max_iterations:
            iterations += 1
            errors = 0
            for i, point in enumerate(data):
                if np.dot(self.w, point) >= 0:
                    if labels[i] == 0:
                        self.update_perceptron(point, labels[i])
                        errors += 1
                        if self.breaking:
                            break
                else:
                    if labels[i] == 1:
                        self.update_perceptron(point, labels[i])
                        errors += 1
                        if self.breaking:
                            break


    def predict_labels(self, data):
        labels = np.zeros(data.shape[0], dtype='int')
        for i, point in enumerate(data):
            labels[i] = (np.dot(self.w, point) >= 0)
        return labels

class Logistic_Regression():
    def __init__(self, w, step_size = 1):
        self.w = w
        self.step = step_size

    def sigmoid(self, data):
        return 1/(1+np.exp(-data))

    def update_classifier(self, data, labels):
        sigmoid_predictions = self.sigmoid(np.dot(data, self.w))
        self.w += self.step*np.dot(data.T, labels - sigmoid_predictions)

    def progress_classifier(self, data, labels, max_iterations):
        iterations = 0
        while iterations < max_iterations:
            self.update_classifier(data, labels)
            iterations += 1

    def predict_labels(self, data):
        labels = np.zeros(data.shape[0], dtype='int')
        for i, point in enumerate(data):
            labels[i] = (np.dot(self.w, point) >= 0)
        return labels

def calculate_error(predicted_labels, labels):
    errors = int(np.sum(1*(predicted_labels != labels), dtype='int'))
    accuracy = (1 - (errors/labels.shape[0]))*100
    return errors, accuracy

def preprocess_text(text):
    text = text.replace("Subject", "")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    return text

df = pd.read_csv('emails.csv')
train_labels = df['spam'].to_numpy()
data = df['text'].apply(preprocess_text)
vectoriser = TfidfVectorizer(stop_words='english')
vectorised_train_data = vectoriser.fit_transform(data)
vectorised_train_data = vectorised_train_data.toarray()
binary_train_data = np.array(1*(vectorised_train_data != 0))

# add code to read the test emails
test_data = []
test_dir_name = 'test'
num_emails = len(os.listdir(test_dir_name))
for i in range(num_emails): # to be changed here
    filename = f'email{i+1}.txt'
    with open(os.path.join(test_dir_name,filename)) as file:
        test_data.append(preprocess_text(file.read()))
vectorised_test_data = vectoriser.transform(test_data)
vectorised_test_data = vectorised_test_data.toarray()
binary_test_data = np.array(1*(vectorised_test_data != 0))

# This code executes the Perceptron Classifier.
w = np.zeros(vectorised_train_data.shape[-1])
breaking = False
Perceptron_Classifier = Perceptron(w, breaking)
num_perceptron_iterations = 10**(1+1*breaking)
Perceptron_Classifier.progess_perceptron(vectorised_train_data, train_labels, num_perceptron_iterations)
Perceptron_pred_labels = Perceptron_Classifier.predict_labels(vectorised_test_data)

# This code executes the Logistic Regression Classifier.
w = np.zeros(vectorised_test_data.shape[-1])
step_size = 0.1
LR_Classifier = Logistic_Regression(w, step_size)
num_LR_iterations = 100
LR_Classifier.progress_classifier(vectorised_train_data, train_labels, num_LR_iterations)
LR_pred_labels = LR_Classifier.predict_labels(vectorised_test_data)

# This code executes the Naive Bayes Classfier
NB_Classifier = Naive_Bayes(vectorised_train_data.shape[-1])
NB_Classifier.update_parameters(binary_train_data, train_labels)
NB_pred_labels = NB_Classifier.predict_labels(binary_test_data)

# This code executes the SVM Classifier
SVM_Classifier = svm.LinearSVC(dual='auto')
SVM_Classifier.fit(vectorised_train_data, train_labels)
SVM_pred_labels = SVM_Classifier.predict(vectorised_test_data)

# This code executes the Mixed Majority Classifier
Best_pred_labels = 1*(np.sum(np.array([Perceptron_pred_labels, LR_pred_labels, NB_pred_labels, SVM_pred_labels]), axis=0) >= 2)

# writing the predicted labels into the csv file
pred_data = list(zip(Perceptron_pred_labels.tolist(), LR_pred_labels.tolist(), NB_pred_labels.tolist(), SVM_pred_labels.tolist(), Best_pred_labels.tolist()))
with open("predicted_labels.csv", 'w') as file:
    headers = ['Perceptron', 'Logistic Regression', 'Naive-Bayes', 'SVM', 'Mixed Majority']
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(pred_data)
