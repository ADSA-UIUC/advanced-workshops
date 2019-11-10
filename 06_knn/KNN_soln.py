import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class KNN:
    def __init__(self, train_features=None, train_labels=None):
        self.train(train_features, train_labels)

    def train(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels

    def classify(self, test_features, k):
        return [self._classify(feature, k) for feature in test_features]

    def _classify(self, feature, k):
        diff = self.train_features - feature
        dists = np.sum(np.square(diff), axis=1)
        k_nearest_indices = np.argpartition(dists, k)[:k]
        k_nearest_labels = Counter(self.train_labels[idx] for idx in k_nearest_indices)

        return k_nearest_labels.most_common(n=1)[0][0]

if __name__ == '__main__':
    # Importing the Iris dataset with pandas
    dataset = pd.read_csv('./iris.csv')
    X = dataset.iloc[:, [1, 2, 3, 4]].values
    y = dataset.iloc[:, 5].values

    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.3, shuffle=True)

    model = KNN(train_features=train_features, train_labels=train_labels)

    expected = model.classify(test_features, 10)
    print(classification_report(test_labels, expected))
