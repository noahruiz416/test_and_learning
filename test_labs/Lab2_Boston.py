#Lab #2 For ITSLR Course
#Author: Noah Ruiz
#Dependent on hydrogen and atom IDE, getting used to basic cleaning and filtering functions

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#a) loading data
#13 columns in this dataset exist, representing features of a house, in the boston area
#our sample size is 505, which represents the 505 houses that we have collected data on
housing = pd.read_csv("/Users/n/StatLearningPy/Data/Boston.csv")
housing = (housing - housing.mean()) / housing.std()

housing = housing.drop(columns = 'Unnamed: 0', index = 1)

housing.info()

#b)histograms, to understand shape + skew of each variable
housing.hist(figsize=(20,10))
pd.plotting.scatter_matrix(housing.iloc[:,0:10], figsize = (15, 10))

housing.info()
#tax and rad appear to be realted to crime
housing.corrwith(housing['crim'])
#though a relationship does exist between rad, tax, indust, lstat and crim it appears to be nonlinear and, I do not think a strucutred model will fit the data very well
pd.plotting.scatter_matrix(housing[['indus', 'rad', 'tax', 'lstat']], figsize = (15, 10))
