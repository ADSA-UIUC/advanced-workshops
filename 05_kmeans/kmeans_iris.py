#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Importing the Iris dataset with pandas
dataset = pd.read_csv('./iris.csv')
x = dataset.iloc[:, [1, 2, 3, 4]].values

plt.scatter(x[:,0], x[:,1])
plt.show()
