import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler




# a=[[[1, 2, 3, 0],[4, 5, 6, 0]],[[1,1,1,1],[2,2,2,2]]]
# print(np.mean(a))
# print(np.std(a))

with open("lstm_single.sh",'w') as f :
    for i in range(2,101):
        for epochs in [50,75,100,150,200]:
            sh_str = "python lstm_single_variable.py --sequence_length " + str(i)+" --epochs "+str(epochs)
            f.write(sh_str)
            f.write("\n")
            f.flush()

#


# for i in range(1,5):
#     print(i)

# x=np.array([[1,2,0],[0,2,0]]).astype(float)
# scaler = MinMaxScaler()
# x_scaler = scaler.fit_transform(x)
# print(x_scaler)
# x_inv=scaler.inverse_transform(x_scaler)
# print(x_inv)
# print(x)

# print(keras.__version__)
# df = pd.read_csv('B0005_cycle_capacity.csv', sep=',')
# print(df.head(5))