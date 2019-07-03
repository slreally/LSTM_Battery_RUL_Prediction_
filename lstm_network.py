from keras.models import Sequential,model_from_json
from keras.layers import Dropout,LSTM,Dense,Activation
import numpy as np

class lstm_network():

    def build_network(self,seq_len,features_num =1,dropout_prob=0.2):
        # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
        model = Sequential()
        
        model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_len, features_num)))
        print(model.layers)
        model.add(Dropout(float(dropout_prob)))
        model.add(LSTM(units=50))
        model.add(Dropout(float(dropout_prob)))
        model.add(Dense(units=1))
        model.add(Activation('linear', name='LSTMActivation'))

        model.compile(loss='mse', optimizer='rmsprop')
        model.summary()
        return model

