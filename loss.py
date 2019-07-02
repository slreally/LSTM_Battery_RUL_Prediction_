import math
import numpy as np
from sklearn.metrics import mean_squared_error
'''
mean square error
'''
def get_rmse(test_y, predict_y):
    mse = math.sqrt(mean_squared_error(test_y,predict_y))
    return mse

'''
mean absolute percentage error
'''
def get_mape(y_true,y_predict):
    y_true,y_predict = np.array(y_true),np.array(y_predict)
    return np.mean(np.abs((y_true-y_predict)/y_true))*100