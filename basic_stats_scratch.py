import numpy as np
import pandas as pd
from math import sqrt

#this function computes basic descriptive stats for a given dataset, used for educational / expereimental purposes

#stats class
#only calculates for one given dataset
class descriptive_stats:
    def __init__(self, univariate_data):
        self.Data = univariate_data
        self.mean = 0
        self.median = 0
        self.variance = 0
        self.std_dev = 0

    #used to calculate the arithmetic mean, uses numpy to speed up computation time
        #takes a data set as input outputs arithmetic mean
    def calculate_mean(self):
        x = self.Data
        n = len(x)
        self.mean = 1/n * np.sum(x)
        return self.mean

    #used to calculate the median, uses numpy to speed up computation time
        #takes a data set as input outputs median
    def calculate_median(self):
        x = self.Data
        n = len(x)

        #median is calculated in two ways depending on whether or not the dataset is even or odd
        #if odd
        if n % 2 != 0:
            self.median = (n + 1) / 2
        #if even
        elif n % 2 == 0:
            left_val = (n/2)
            right_val = (n/2) + 1
            self.median = ((left_val + right_val) / 2)
        #error statement for debugging
        else:
            print("Error")

        return self.median

    #used to calculate the variance in a given dataset
        #takes in a set of data as input and outputs variance, depends on the mean function to compute var
    def calculate_variance(self):
        x = self.Data
        n = len(x)
        mean = self.calculate_mean()
        self.variance = np.sum((x - mean)**2) / (n-1)
        return self.variance

    #used to calc std dev of a data set
        #simply take sqrt of variance
    def calculate_std_dev(self):
        x = self.Data
        var = self.calculate_variance()
        self.std_dev = sqrt(var)
        return self.std_dev

    #this function calls the functions above and is used to give the user an output of essential descriptive stats
    def return_stats(self):
        mean = self.calculate_mean()
        median = self.calculate_median()
        variance = self.calculate_variance()
        std_dev = self.calculate_std_dev()

        vals = {'Mean': round(mean,4),
                'Median': median,
                'Variance': round(variance,4),
                'Std Dev': round(std_dev,4)}
        display(vals)

def main():

    auto_data = pd.read_csv("/Users/n/StatLearningPy/Data/Auto.csv")
    stats = descriptive_stats(auto_data['displacement'])
    stats.return_stats()

main()
