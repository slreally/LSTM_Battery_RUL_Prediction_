from keras.models import Sequential,model_from_json
from keras.layers import Dropout,LSTM,Dense,Activation
import numpy as np

#TODO:可优化成两个类似生成器的东西，一个自定义模型，一个通过已有模型导入
class lstm():
    def __init__(self,seq_len,features_num,dropout_prob,units_num=50):
        self.seq_len = seq_len
        self.features_num = features_num
        self.dropout_prob = dropout_prob
        self.units_nums = units_num


    def build_model(self):
        # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
        model = Sequential()
        model.add(LSTM(units= 50, return_sequences=True, input_shape=(self.seq_len, self.features_num)))
        print(model.layers)
        model.add(Dropout(float(self.dropout_prob)))
        model.add(LSTM(units= 50))
        model.add(Dropout(float(self.dropout_prob)))
        model.add(Dense(units=1))
        model.add(Activation('linear', name='LSTMActivation'))

        model.compile(loss='mse', optimizer='rmsprop')
        model.summary()
        return model

    def train_model(self,model,train_x, train_y, batch_size, epochs):

        try:
            model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True,
                      verbose=1)

        except KeyboardInterrupt:
            print(" train model error")

        return model

    def predict(self,model,test_x,pre_way):
        predict = self.predict_way(model,test_x,way=pre_way)
        return predict


    '''
    way = one_cycle:predict one cycle.
    else:predict len(test_x) cycles.use previous prediction value as current input x.
    '''
    def predict_way(self,model, predict_data, way=0):
        if way == 0:
            predict = model.predict(predict_data)
            return np.reshape(predict, (predict.size, 1))
        else:
            predict = self.predict_sequences_multiple(model, predict_data, len(predict_data))
            return np.reshape(predict, (-1, 1))

    # 电池预测cycle时的capacity,输入应是cycle-1时的预测值capacity
    def predict_sequences_multiple(self,model, data, prediction_len):
        # Predict sequence of pre_len steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        data_begin = data[0, :, :]
        data_begin = np.reshape(data_begin, (1, data_begin.shape[0], data_begin.shape[1]))
        prediction_seqs.append(model.predict(data_begin))
        for i in range(1, prediction_len):
            data_begin = data_begin[0, 1:, :]
            data_i = (data[i, -1:, :]).copy()
            data_i[-1][-1] = prediction_seqs[-1]
            data_begin = np.concatenate((data_begin, data_i), axis=0)
            data_begin = np.reshape(data_begin, (1, data_begin.shape[0], data_begin.shape[1]))
            prediction_seqs.append(model.predict(data_begin))

        return prediction_seqs

    # json_filepath: 'multi_battery/batch_size:32_epochs:50_premeas:0.json'
    # model_weight_filepath:'multi_battery/batch_size:32_epochs:50_premeas:0.h5'
    def load_model(self,json_filepath,model_weight_filepath):
        json_file = open(json_filepath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weight_filepath)
        print('loaded model from disk')
        loaded_model.compile(loss='mse', optimizer='rmsprop')
        return loaded_model


