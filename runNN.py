from sklearn.neural_network import BernoulliRBM
import os
from sklearn.externals import joblib
import numpy as np
import csv
import cv2
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn import neighbors
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import entropy
from collections import Counter
from matplotlib import pyplot as plt



X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

model = BernoulliRBM(n_components=2)

model.fit(X)
BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=2, n_iter=10, random_state=None, verbose=0)