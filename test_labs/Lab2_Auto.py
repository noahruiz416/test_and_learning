#Lab #2 For ITSLR Course
#Author: Noah Ruiz
#Dependent on hydrogen and atom IDE, getting used to basic cleaning and filtering functions

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#a) loading data
auto_data = pd.read_csv("/Users/n/StatLearningPy/Data/Auto.csv")

#checking for null values
auto_data.isnull().sum()

#checking quantitative variables
auto_data.select_dtypes(exclude=["number","bool_","object_"])

#9a) getting just the qualitative variables and quantiative
numeric_data = auto_data.select_dtypes(include=[np.number])
categorical_data = auto_data.select_dtypes(exclude=[np.number])

#9b) find range of each quantiative variables
range_numeric = numeric_data.max() - numeric_data.min()
range_numeric

#9c) find mean and std for each quantiative varaible
std_devs = numeric_data.std()
mean_numeric = numeric_data.mean()

#9d) remove 10th through 85th observation
filter_auto = auto_data.drop(labels=auto_data.index[9:85], axis = 0)
filter_auto

#9e) Basic Exploratory Analysis
#based on the scatter matrix it appears that a curvilinear / quadratic relationship does exist between displacelment and weight
#additionally a curvilinear strong negative relationsjip also exists between mpg and weight as well as mpg and displacment
#further almost all of our varaibles seem to have skewed distributions to the right, with the excepction of accelearation which follows a relatively normal distribution
#this indicates that normalization may need to be conducted to see if our dataset follows a gaussian distribution
pd.plotting.scatter_matrix(numeric_data, figsize = (10, 10))

#9f) Predicting gas mileage, which features to use?
#first use correlation matrix to find any inherent relationships
auto_data.corrwith(auto_data['mpg'])

auto_data


#using a regplot to viz possible x variables
#displacemnet and weight seem to be the strongest canidates for potential variables, though year and accelearation appear promising as well
#it may help to add quadratic terms to a potential model
fig, axs = plt.subplots(ncols=5, figsize = (25, 5))
sns.regplot(x='cylinders', y='mpg', data=auto_data, ax=axs[0])
sns.regplot(x='displacement', y='mpg', data=auto_data, ax=axs[1])
sns.regplot(x='weight',y='mpg', data=auto_data, ax=axs[2])
sns.regplot(x='year',y='mpg', data=auto_data, ax=axs[3])
sns.regplot(x='acceleration',y='mpg', data=auto_data, ax=axs[4])
