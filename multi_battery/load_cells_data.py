import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation,Dropout,Flatten
import datetime
import random
import math
import codecs
import os.path as osp
from numpy import newaxis
import argparse

class load_cells_data():
    def __init__(self,filename,seq_len,split,usecols =[0,1]):
        self.file_name = filename
        self.sequence_length = seq_len
        self.split = split
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.df = pd.read_csv(self.file_name, sep=',', usecols=usecols)
        self.data_all = np.array(self.df).astype(float)

    def get_x_y(self):
        train_x,train_y =self.get_train_x_y(self.data_all)
        return train_x,train_y

    def get_train_x_y(self, data_train):
        data_scalered = self.scaler_train_data(data_train)
        data = []
        for i in range(len(data_scalered) - self.sequence_length + 1):
            # data.append(data_scalered[i: i + self.sequence_length - 1])
            data.append(data_scalered[i: i + self.sequence_length ])
        reshaped_data = np.array(data).astype('float64')

        random.shuffle(reshaped_data)

        train_x = reshaped_data[:, :, :-1]
        train_y = reshaped_data[:, len(reshaped_data[1]) - 1, -1:]##-2:-1
        return train_x, train_y

    def get_test_x_y(self,data_test):
        data_scalered = self.scaler_test_data(data_test)
        data = []
        for i in range(len(data_scalered) - self.sequence_length + 1):
            # data.append(data_scalered[i: i + self.sequence_length - 1])
            data.append(data_scalered[i: i + self.sequence_length ])
        reshaped_data = np.array(data).astype('float64')

        test_x = reshaped_data[:, :, :-1]
        test_y = reshaped_data[:, len(reshaped_data[1]) - 1, -1:]
        return test_x, test_y

    def scaler_train_data(self,data):
        data_x = self.scaler_x.fit_transform(data[:, :-1])
        data_y = self.scaler_y.fit_transform(data[:, -1:]) #label不归一化
        # data_y =data[:,-1:]
        data_all = np.concatenate((data_x, data_y), axis=1)
        return data_all

    def scaler_test_data(self,data_test):
        data_x = self.scaler_x.transform(data_test[:,:-1])
        data_y = self.scaler_y.transform(data_test[:,-1:])
        # data_y = data_test[:, -1:]
        data_all = np.concatenate((data_x,data_y),axis=1)
        return data_all

    def get_scaler_x_y(self):
        return self.scaler_x,self.scaler_y

    def get_all_y(self):
        return  self.data_all[:,-1:]


def main():
    dataloader =load_cells_data("2017_06_30_cell0_data.csv",20,0.5)


if __name__ == '__main__':
    main()