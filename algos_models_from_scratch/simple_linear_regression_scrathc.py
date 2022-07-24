import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

#simple class to calculate mse and apply gradient descent to a simple linear regression problem
#this is simply done for learning and to gain a better understanding of linear regression computed algorithmically

#this class takes in and input vector X and requires a target variable y
#somewhat works, however have issues choosing optimal learning rate
class simple_linear_regression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.theta = [0,0]

    def gradient_descent(self, learning_rate):
        pred_y_vals = self.make_prediction(self.X)
        n = len(self.X)
        self.theta[0] = self.theta[0] - ((learning_rate/n) * np.sum(pred_y_vals - self.y))
        self.theta[1] = self.theta[1] - ((learning_rate/n) * np.sum((pred_y_vals -self.y) * self.X))
        return self.theta

    def make_prediction(self, input_vector):
        pred_vals = np.array([])
        for x in input_vector:
            pred_vals = np.append(pred_vals, self.theta[0] + self.theta[1] * x)
        return pred_vals


    def cost_function_mse(self, pred_y_values):
        n = len(self.y)
        err = (1/n) * np.sum((self.y - pred_y_values)**2)
        return err

    #getter
    def get_params(self):
        print(self.theta[0])
        print(self.theta[1])

#testing cost_function & gradient descent to estimate params
#generating test data
def main():
    input = np.array([1, 2, 2, 3])
    target = np.array([3, 4, 5, 6])
    linear_reg = simple_linear_regression(input, target)

    #using gradient descent to estimate parameters
    linear_reg.gradient_descent(0.182)

    #gets params
    #linear_reg.get_params()

    #using updated function to make predictions
    preds = linear_reg.make_prediction(input)

    display(preds)
    display(linear_reg.cost_function_mse(preds))

    plt.scatter(input, target, color='b')
    plt.plot(input, preds, color='g')
    plt.show()

main()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


X = np.array([1, 2, 2, 3])
y = np.array([3, 4, 5, 6])

X = X.reshape(-1,1)

reg = LinearRegression().fit(X, y)

pred = (reg.predict(X))

mean_squared_error(y, pred)
