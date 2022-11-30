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

# Trying this section out before moving forward with factorising code into functions.....

loss_function_list = ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]
for loss_function in loss_function_list:
    #reg = make_pipeline(StandardScaler(), SGDRegressor(loss = loss_function, max_iter= 10000000, tol=1e-8,eta0=0.0001,power_t=0.3, learning_rate = 'optimal'))
    model = TabularDataClean()
    train = model.load_airbnb('clean_tabular_data.csv')
    X,y=train
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
    ss=StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    y_train = np.ravel(y_train)
    fitted_data = SGDRegressor(loss = loss_function, max_iter= 10000000, tol=1e-8,eta0=0.0001,power_t=0.3, learning_rate = 'optimal').fit(X_train,y_train)
    y_hat=fitted_data.predict(X_test)
    joblib.dump(model, f"model_{loss_function}.joblib")

    plt.figure()

    samples1= len(y_hat)
    samples2 = len(y_test)
    plt.scatter(np.arange(samples1),y_hat, c='red', label = 'fitted data')
    plt.scatter(np.arange(samples2),np.ravel(y_test),c='blue',label= 'true data')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

    print(f"The fitted score date for {loss_function} model is:"+str(fitted_data.score(X,y)))

    y_hat_flat= np.ravel(y_hat)
    y_test_flat = np.ravel(y_test)
    print()
    print()
    print()
    print()
    print(f"Error Values for model using : {loss_function}")
    print("MSE (Python):", MSE(y_test_flat, y_hat))
    print("MSE (scikit-learn):", mean_squared_error(y_test_flat, y_hat))
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
