import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv("/Users/n/StatLearningPy/Data/Hitters.csv")

df.dropna(inplace = True)

#drops salary from input
input_vector = df.drop(labels = ['Salary'], axis = 1)
X = input_vector

#collecting target data
y = df['Salary']

#now we will preprocess the data, with first a scale then OHE
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
object = ['object']

#setting up pipelines

#gets numeric features in dataset
numeric_features = X.select_dtypes(include=numerics).columns
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

#categorical pipeline
categorical_features = X.select_dtypes(include=object).columns
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

#preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

#creating our pipeline for linear regression
linear_regression = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', LinearRegression())])

#getting train test split ready
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .20)

#fitting the dataset
linear_regression.fit(x_train, y_train)

print("model score: %.3f" % linear_regression.score(x_test, y_test))

from sklearn.metrics import mean_squared_error
y_pred = linear_regression.predict(x_test)
y_true = y_test
mean_squared_error(y_true, y_pred)

#now we will fit a lasso regression to the data to see if performance improves
from sklearn.linear_model import LassoCV
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .20)
lasso_regression = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', LassoCV(cv = 5))])

lasso_regression.fit(x_train, y_train)
print("model score: %.3f" % lasso_regression.score(x_test, y_test))

#now we will try to fit a neural netwrok as opposed to a regression model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# define base model
def baseline_model():
    model = Sequential()
    model.add(Dense(50, input_shape=(22,), activation = 'relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasRegressor

X = pd.get_dummies(X)

kfold = KFold(n_splits=5)
estimator = KerasRegressor(model=baseline_model(), epochs=100, batch_size=5, verbose=0)
results = cross_val_score(estimator, X, y, cv=kfold, scoring='neg_mean_squared_error')
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
