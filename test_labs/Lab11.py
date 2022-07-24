import pandas as pd
import numpy as np
import seaborn as sns

#a) loading data
data = pd.read_csv("/Users/n/StatLearningPy/Data/BrainCancer.csv")

data.dropna(inplace = True)

data.info()
#aftering getting an idea of our dataset now we can perform survival analysis
from sksurv.nonparametric import kaplan_meier_estimator
import matplotlib.pyplot as plt
data['status_bool'] = data['status'].astype(bool)


time, survival_prob = kaplan_meier_estimator(data["status_bool"], data["time"])
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.step(time, survival_prob, where="post")

#log rank test for gender, T1 males, T2 for females
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

data['sex_binary'] = lb.fit_transform(data['sex'])


Male = data.query(f"sex_binary == {1}")
Female = data.query(f"sex_binary == {0}")

T = Male['time']
E = Male['status_bool']

T1 = Female['time']
E1 = Female['status_bool']

from lifelines.statistics import logrank_test
results = logrank_test(T, T1, event_observed_A=E, event_observed_B=E1, )

#showing there is not a signficant diff between males and females
results.print_summary()

print(results.p_value)

print(results.test_statistic)

#fitting the cox proportional hazards model
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import train_test_split

X_input = data.drop(columns = ['Unnamed: 0', 'time', 'status_bool', 'status', 'sex'], axis =1)
X_input = pd.get_dummies(X_input)
y = data[["status_bool", "time"]]

x_train, x_test, y_train, y_test = train_test_split(X_input,y ,test_size = .20)
from sksurv.util import Surv
y_train_struc = Surv.from_arrays(y_train['status_bool'], y_train['time'])


estimator = CoxPHSurvivalAnalysis(alpha = 0.01).fit(x_train, y_train_struc)

results = pd.Series(estimator.coef_, index=x_train.columns)
print(results)

#different but better library
from lifelines import CoxPHFitter
cph = CoxPHFitter(alpha=0.05, penalizer=1e-5)

X_input = data.drop(columns = ['Unnamed: 0', 'time', 'status_bool', 'status', 'sex'], axis =1)
X_input = pd.get_dummies(X_input)
y = data[["status", "time"]]


x_train, x_test, y_train, y_test = train_test_split(X_input,y ,test_size = .20)

merge = pd.concat([x_train,y_train], axis = 1)

cph.fit(merge, duration_col= 'time', event_col='status')

cph.print_summary()

cph.plot()
