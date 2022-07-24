#Lab #2 For ITSLR Course
#Author: Noah Ruiz
#Dependent on hydrogen and atom IDE, getting used to basic cleaning and filtering functions

import pandas as pd
import numpy as np
import seaborn as sns

#a) loading data
data = pd.read_csv("/Users/n/StatLearningPy/Data/College.csv")

#b) viewing data + remove "Unnamed: 0 "
data

data = data.drop(columns = "Unnamed: 0", index = 1)

#c.i) getting summary of variables in dataset
data.info()

#c.ii) scatterplot matrix , with first 10 columns, using iloc to subset by index
pd.plotting.scatter_matrix(data.iloc[:,0:10], figsize = (15, 10))

#c.iii) boxplot of Outstate vs Private
data.iloc[:,0:10].boxplot(figsize=(12,10))

#c.v) creating histograms for various quantitative vars
data.iloc[:,0:10].hist(figsize = (12, 10))
