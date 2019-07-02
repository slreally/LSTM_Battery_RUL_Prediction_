import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential,model_from_json
from keras.layers import LSTM, Dense, Activation,Dropout,Flatten
import datetime
import random
import math
import codecs
import os.path as osp
from numpy import newaxis
import argparse

def build_model(seq_len,features_num,dropout_prob=0.5,units_num=50):
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(units=units_num,return_sequences=True,input_shape=(seq_len,features_num)))
    print(model.layers)
    model.add(Dropout(float(dropout_prob)))
    model.add(LSTM(units=units_num))
    model.add(Dropout(float(dropout_prob)))
    model.add(Dense(units=1))
    model.add(Activation('linear',name='LSTMActivation'))

    model.compile(loss='mse', optimizer='rmsprop',metrics=['mae','acc'])
    model.summary()

    return model


def train_model(train_x, train_y, batch_size,epochs,pre_way,dropout):
    model = build_model(train_x.shape[1],train_x.shape[2],dropout_prob=dropout)

    try:
        model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs, validation_split=0.1,shuffle=False,verbose=1)
        scores = model.evaluate(train_x,train_y,verbose=1)
        # print('{0} = {1}'.format(model.metrics_names[1],scores[1]))
        for x in range(len(model.metrics_names)):
            print('{0} = {1}'.format(model.metrics_names[x], scores[x]))

        model_path = 'batch_size:{0}_epochs:{1}_premeas:{2}'.format(str(batch_size), str(epochs), str(pre_way))
        model_json = model.to_json()
        with open(model_path+".json","w") as f:
            f.write(model_json)
        model.save_weights(model_path+'.h5')
    except KeyboardInterrupt:
        print('error')

def main():

    parser = argparse.ArgumentParser(description='LSTM RUL Prediction')
    parser.add_argument('--filename', type=str, default="cells_single_input")
    parser.add_argument('--output_path',type=str,default="snapshot/multi_variable")
    parser.add_argument('--predict_measure', type=int, default=0, choices=[0,1])
    parser.add_argument('--sequence_length', type=int,default=20)
    parser.add_argument('--split', default=0.5, help='split of train and test set')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--dropout', default=0.2)

    args = parser.parse_args()

    split = args.split
    dropout = args.dropout
    sequence_length = args.sequence_length
    batch_size = args.batch_size
    epochs = args.epochs
    predict_measure = args.predict_measure  # 0 for predicting one cycle,1 for predicting len(test(y)) cycles continuely, use current predicted value as the next input.
    filename = args.filename


    import multi_battery.load_cells_data as load_data

    dataloader = load_data.load_cells_data(filename + ".csv", sequence_length, split)
    train_x, train_y = dataloader.get_x_y()

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

    train_model(train_x, train_y, batch_size, epochs, predict_measure, dropout)



if __name__ == '__main__':

    main()