# -*- coding: utf-8 -*-
"""
As part of the UIUC Association of Data Science and Analytics Advanced Workshop Series
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)


class LinearRegression:

    def __init__(self):
        self.coefficients = 0

    def _cost(features, targets, coefficients):
        """cost function to compute the error in our model

        features - numpy matrix of all X variables
        targets - numpy matrix of all Y variables
        coefficients - numpy matrix of the weights of each Xi in X
        return - the cost (error) in our model as a floating point value
        """
        hypothesis = features.dot(coefficients)
        loss = hypothesis - targets
        num_rows = len(targets)
        cost = np.sum(loss ** 2) / (2 * num_rows)
        return cost

    def train(self, X_train, y_train, learning_rate=.0001, iterations=100000):
        """training model for gradient descent multivariate linear regression

        X_train - numpy matrix of all X variables
        y_train - 1d numpy matrix of all Y variables
        learning_rate - the gradient descent step size
        iterations - the number of times the model recalculates its coefficients
        return - the final coefficients as an array
        return - an array of costs from each iteration
        """
        cost_history = []  # Initialize list of size iterations to all 0's
        num_rows = len(y_train)  # Compute number of data points
        self.coefficients = np.zeros(X_train.shape[1])

        for iteration in range(iterations):
            # Compute the predicted target values as a matrix
            hypothesis = X_train.dot(self.coefficients)

            # Compute the difference between hypothesis (predicted) and target (actual) values in the matrices
            loss = hypothesis - y_train

            # Compute the new gradient for this iteration as a float array
            gradient = X_train.T.dot(loss) / num_rows

            # Compute the new coefficients as a matrix from the new gradient
            self.coefficients -= learning_rate * gradient

            # Calculate the cost from this iteration
            cost = np.sum(loss ** 2) / (2 * num_rows)

            # Store the cost in the history
            cost_history.append(cost)

        return cost_history

    def predict(self, X):
        return X.dot(self.coefficients)

    def test(self, X_test, y_test):
        MSE = np.sum([(self.predict(X) - y) ** 2 for X,
                      y in zip(X_test, y_test)]) / len(y_test)
        RMSE = np.sqrt(MSE)

        R2_accuracy = 1 - (MSE / np.var(y_test))

        return RMSE, R2_accuracy


data = pd.read_csv("../student.csv")

# Ploting the scores as scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data['Math'].values, data['Reading'].values,
           data['Writing'].values, color='#ef1234', label='Scatter Plot')
ax.set_xlabel('Math score (%)')
ax.set_ylabel('Reading score (%)')
ax.set_zlabel('Writing score (%)')
plt.title("Reading and Math vs. Writing Scores Data")
plt.legend()
plt.show()

train_data = data.head(700)

math_train = train_data['Math'].values
read_train = train_data['Reading'].values
write_train = train_data['Writing'].values

intercepts = np.ones(len(math_train))
features = np.array([intercepts, math_train, read_train]).T
targets = np.array(write_train)
coefficients = np.array([0, 0, 0])

lin_model = LinearRegression()

cost_history = lin_model.train(features, targets)
new_coefficients = lin_model.coefficients
z_int = new_coefficients[0]
slope_x = new_coefficients[1]
slope_y = new_coefficients[2]

print(
    "z-intercept: {0}, x_slope: {1}, y_slope: {2}".format(z_int, slope_x, slope_y))

print("The cost per 10,000 iterations is: {0}".format(cost_history[::10000]))

# Ploting Values and Regression Line
max_x = np.max(data['Math'].values)
min_x = np.min(data['Math'].values) - 5

max_y = np.max(data['Reading'].values)
min_y = np.min(data['Reading'].values) - 5

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = np.linspace(min_y, max_y, 1000)
z = z_int + slope_x * x + slope_y * y

# Ploting Line

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data['Math'].values, data['Reading'].values,
           data['Writing'].values, color='#ef1234', label='Scatter Plot')
ax.plot(x, y, z, color='#58b970', label='Regression Line')
ax.set_xlabel('Math score (%)')
ax.set_ylabel('Reading score (%)')
ax.set_zlabel('Writing score (%)')
plt.title("Reading and Math vs. Writing Scores Data")
plt.legend()
plt.show()

test_data = data.tail(300)

math_test = test_data['Math'].values
read_test = test_data['Reading'].values
write_test = test_data['Writing'].values

test_intercepts = np.ones(len(math_test))
test_features = np.array([test_intercepts, math_test, read_test]).T
test_targets = np.array(write_test)


rmse, R2_score = lin_model.test(test_features, test_targets)
print("Root Mean Square Error (RMSE): {0}, R^2 Score: {1}".format(
    rmse, R2_score))
