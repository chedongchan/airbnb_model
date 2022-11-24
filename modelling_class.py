# %%

from tabular_data_class import TabularDataClean
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


import joblib
import numpy as np
import matplotlib.pyplot as plt

def MSE(targets, predicted):
    return np.mean(np.square(targets - predicted))

def RMSE(targets, predicted):
    return np.sqrt(MSE(targets, predicted))

def MAE(targets, predicted):
    return np.mean(np.abs(targets - predicted))

def R2(targets, predicted):
    return 1 - (MSE(targets, predicted)/np.var(targets))

reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter= 10000000, tol=1e-8,eta0=0.000001,power_t=0.5))
model = TabularDataClean()
train = model.load_airbnb('clean_tabular_data.csv')
X,y=train
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
y = np.ravel(y)
fitted_data = reg.fit(X,y)
y_pred=reg.predict(X)
joblib.dump(model, "model.joblib")

plt.figure()

samples1= len(y_pred)
samples2 = len(y_test)
plt.scatter(np.arange(samples1),y_pred, c='red', label = 'fitted data')
plt.scatter(np.arange(samples2),np.ravel(y_test),c='blue',label= 'true data')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.show()

print(fitted_data.score(X,y))

y_train_flat = np.ravel(y_train)
my_linear_model = SGDRegressor(eta0=0.000001,power_t=0.5).fit(X_train,y_train_flat)
y_hat = my_linear_model.predict(X_test)
y_hat_flat= np.ravel(y_hat)
y_test_flat = np.ravel(y_test)

print("MSE (Python):", MSE(y_test_flat, y_hat))
print("MSE (scikit-learn):", mean_squared_error(y_test_flat, y_hat))
print()
print("RMSE (Python):", RMSE(y_test_flat, y_hat))
print("RMSE (scikit-learn):", mean_squared_error(y_test_flat, y_hat, squared=False))

# Computing MAE for our Linear Model
print("MAE (Python):", MAE(y_test_flat, y_hat))
print("MAE (scikit-learn):", mean_absolute_error(y_test_flat, y_hat))

# Computing R2
print("R2 (Python):", R2(y_test_flat, y_hat))
print("R2 (scikit-learn):", r2_score(y_test_flat, y_hat))


def custom_tune_regression_model_hyperparameters(model_class, test_set_list,hyperparameters):
    pass
