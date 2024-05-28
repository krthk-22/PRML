# PRML
This repo contains assignments of Pattern Recognition and Machine Learning(PRML) course under Prof. Arun Raj Kumar during Jan 2024 - May 2024

## PCA - Clustering

The [PCA-Clustering](/PCA-Clustering/) folder contains the following.
1. **PCA** - Principal Component Analysis
    - PCA is applied on ***mnist*** dataset.
    - Kernel version of PCA is applied on ***mnist*** dataset.

2. **Clustering** - Following Clustering Algorithms are applied and their performance is compared on the ***Cresent-Moon*** dataset.
    - K-means
    - Kernel K-means
    - Spectral Clustering

## EM - Regression
The [EM-Regression](/EM-Regression/) folder contains the following.
1. **EM-Algorithm**
    - Data is given our goal to find the distribution that is used to generate the given data.
    - This can be done by maximising likelihood of the data under our model
    - Tried implementing the Bernoulli and Gaussian variants of the EM algorithm along with K-means.

2. **Regression**
    - Best fit line is found for the given data using Linear-Regression
    - The best fit line is found using different methods like
        - Analytical Method
        - Gradient Descent
        - Stochastic Gradient Descent
        - Ridge-Regression

## Classification
The [Classification](/Classification/) folder contains the following.
1. **Spam vs Non-Spam** email classifier
    - A ~simple~ *Spam vs Non-Spam* email classifier is created.
    - The following classification methods are used in creating the classifier.
        - Bernoulli Naive-Bayes
        - Gaussian Naive-Bayes
        - Perceptorn
        - Logistic Regression
        - Support Vector Machine (SVM)