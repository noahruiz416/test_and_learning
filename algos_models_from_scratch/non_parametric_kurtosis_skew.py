#for data pulling and implementation
import pandas as pd
import numpy as np
from math import sqrt

#for testing handmade functions
import scipy.stats as stats
from scipy.stats import kurtosis, skew

#this class computes the kurtosis and skew of a variable
class kurotsis_skew:
    def __init__(self, X):
        self.x = X
        self.n = len(X)
    def mean(self):
        #sum vars * 1 / n-1

        mean = (1/(self.n-1)) * (np.sum(self.x))
        return mean
    def std_dev(self):
        #(summation (x - mu)^2) / (n-1)

        mean = self.mean()
        variance = np.sum((x - mean)**2) / (self.n-1)
        std_dev = sqrt(variance)
        return std_dev

    #kurtosis checks how flat the top of a distribution is with respect to a normal distirbution
    #negative values indicate a flat top
    #positive a peaked top
    #zero a perfectly normal dist

    def approx_kurtosis(self):
        mean = self.mean()
        std_dev = self.std_dev()
        #calculating kurtosis
        #num is fourth central moment, den standard dev

        kurtosis_num = np.sum((self.x - mean)**4)
        kurtosis_den = std_dev ** 4
        kurtosis = (1/self.n )* (kurtosis_num / kurtosis_den)

        #correction to standardize kurtosis around the normal dist since we assume 3 std devs

        return kurtosis - 3

    def approx_skew(self):
        mean = self.mean()
        std_dev = self.std_dev()

        #calculating skew
        #we can define skew as the third moment of a random variable
        #skewness can be negative or positive
        #negative indicates left skewed dist
        #positie indicates right skewed dist

        skew_num = np.sum((self.x - mean)**3)
        skew_den = (self.n - 1) * std_dev**3
        skew = (skew_num / skew_den)
        return skew

#testing our class
df = pd.read_csv("/Users/n/StatLearningPy/Data/Auto.csv")

x = df['displacement']

kurt_skew = kurotsis_skew(x)

kurt_skew.mean()

kurt_skew.std_dev()

kurt_skew.approx_kurtosis()

kurt_skew.approx_skew()

#testing skew and kurtosis

kurtosis(x)

skew(x)

x.hist()
