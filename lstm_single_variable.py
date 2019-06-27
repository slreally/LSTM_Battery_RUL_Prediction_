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


def load_data(file_name, sequence_length=50, split=0.3):
    df = pd.read_csv(file_name, sep=',', usecols=[9,10])

    data_all = np.array(df).astype(float)

    scaler_x = MinMaxScaler()
    data_x = scaler_x.fit_transform(data_all[:,:-1])
    # scaler_y = MinMaxScaler()
    # data_y = scaler_y.fit_transform(data_all[:,-1:])
    data_all = np.concatenate((data_x,data_all[:,-1:]),axis=1)

    data = []
    for i in range(len(data_all) - sequence_length + 1):
        data.append(data_all[i: i + sequence_length])
    reshaped_data = np.array(data).astype('float64')
    # np.random.shuffle(reshaped_data)
    # 对x进行统一归一化，而y则不归一化
    x = reshaped_data[:, :,:-1]
    y = reshaped_data[:, len(reshaped_data[1])-1,-1:]
    split_boundary = int(reshaped_data.shape[0] * split)

    train_x = x[: split_boundary]
    test_x = x[split_boundary:]

    train_y = y[: split_boundary]
    test_y = y[split_boundary:]

    # fig = plt.figure(1)
    # plt.plot(y)
    # plt.show()

    return data_all,y,split_boundary,train_x, train_y, test_x, test_y, scaler_x#,scaler_y


def build_model(seq_len,features_num,dropout_prob=0.2,units_num=50):
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(units=units_num,return_sequences=True,input_shape=(seq_len,features_num)))
    print(model.layers)
    model.add(Dropout(float(dropout_prob)))
    model.add(LSTM(units=units_num))
    model.add(Dropout(float(dropout_prob)))
    model.add(Dense(units=1))
    model.add(Activation('linear',name='LSTMActivation'))

    model.compile(loss='mse', optimizer='rmsprop')
    model.summary()

    return model


def train_model(train_x, train_y, test_x,batch_size,epochs,pre_way,dropout):
    model = build_model(train_x.shape[1],train_x.shape[2],dropout_prob=dropout)

    try:
        model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs, validation_split=0.1,shuffle=False,verbose=2)
        predict = predict_way(model,test_x,way=pre_way)
    except KeyboardInterrupt:
        print(predict)

    return predict

'''
way = one_cycle:predict one cycle.
else:predict len(test_x) cycles.use previous prediction value as current input x.
'''
def predict_way(model, predict_data, way=0):
    if way == 0:
        predict = model.predict(predict_data)
        return np.reshape(predict, (predict.size,1))
    else:
        predict = predict_sequences_multiple(model, predict_data, len(predict_data))
        return np.reshape(predict, (-1,1))


#电池预测cycle时的capacity,输入应是cycle-1时的预测值capacity
def predict_sequences_multiple(model,data, prediction_len):
    # Predict sequence of pre_len steps
    print('[Model] Predicting Sequences Multiple...')
    prediction_seqs = []
    data_begin = data[0, :, :]
    data_begin = np.reshape(data_begin,(1,data_begin.shape[0],data_begin.shape[1]))
    prediction_seqs.append(model.predict(data_begin))
    for i in range(1,prediction_len):
        data_begin = data_begin[0,1:,:]
        data_i =(data[i,-1:,:]).copy()
        data_i[-1][-1] = prediction_seqs[-1]
        data_begin = np.concatenate((data_begin,data_i),axis=0)
        data_begin = np.reshape(data_begin, (1, data_begin.shape[0], data_begin.shape[1]))
        prediction_seqs.append(model.predict(data_begin))

    # for i in range(int(len(data) / prediction_len)):
    #     curr_frame = data[i * prediction_len]
    #     predicted = []
    #     for j in range(prediction_len):
    #         predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
    #         curr_frame = curr_frame[1:]
    #         curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
    #     prediction_seqs.append(predicted)
    return prediction_seqs

def plot_and_save(title,sequence_length,filename,all_y,test_x,test_y,train_y,predict_y):
    fig2 = plt.figure()
    plt.xlabel('cycle')
    plt.ylabel('QD')
    # plt.plot(range(sequence_length,sequence_length+len(all_y)),all_y)
    plt.plot(all_y)
    plt.plot(range(sequence_length,sequence_length+len(train_y),1),train_y,'m:')
    plt.plot(range(sequence_length+len(train_y),sequence_length+len(train_y)+len(test_y),1),test_y,'r:')
    plt.plot(range(sequence_length+len(train_y),sequence_length+len(train_y)+len(predict_y),1),predict_y,'g-')
    time = datetime.datetime.now().strftime('%m-%d-%H-%R-%S')
    plt.title(title)
    plt.legend(['ground truth','train','test','predict'])
    # plt.show()
    filename =filename+str(time)
    plt.savefig('result/single_variable/'+filename+'.png')
    plt.close(fig2)

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

def main():

    parser = argparse.ArgumentParser(description='LSTM RUL Prediction')
    parser.add_argument('--filename', type=str, default="2017_06_30_cell0_data")
    parser.add_argument('--output_path',type=str,default="snapshot/single_variable")
    parser.add_argument('--predict_measure', type=int, default=0, choices=[0,1])
    parser.add_argument('--sequence_length', type=int,default=9)
    parser.add_argument('--split', default=0.5, help='split of train and test set')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
    output_path = args.output_path
    file_error = 'seqlen:'+str(sequence_length)+'_mse_mape'+str(datetime.datetime.utcnow())+'.txt'

    fo = open(osp.join(output_path,file_error),'w')
    fo.write(str('N,batch_size,epochs,mse,mape\n'))
    fo.flush()

    batch_size_list = [8,16,32,64,128]
    epochs_list = [50,75,100,150,200]

    import load_data
    for batch_size in batch_size_list:
        dataloader = load_data.load_data(filename + ".csv", sequence_length, split,usecols=[9,10])
        train_x, train_y, test_x, test_y = dataloader.get_x_y()
        all_y = dataloader.get_all_y()

        # data,all_y,split_boun,train_x,train_y,test_x,test_y,scal_x= load_data(filename+".csv",sequence_length,split)

        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
        print(train_y.shape)
        predict_y = train_model(train_x, train_y, test_x, batch_size, epochs, predict_measure, dropout)

        sca_x,sca_y = dataloader.get_scaler_x_y()
        train_y = sca_y.inverse_transform(train_y)
        test_y = sca_y.inverse_transform(test_y)
        predict_y = sca_y.inverse_transform(predict_y)
        mse = get_rmse(test_y, predict_y)
        mape = get_mape(test_y, predict_y)

        err_str = '{0},{1},{2},{3},{4}\n'.format(sequence_length, batch_size, epochs, mse, mape)
        fo.write(str(err_str))
        fo.flush()

        # all_y = scaler_y.inverse_transform(all_y)
        # train_y = scaler_y.inverse_transform(train_y)
        # test_y = scaler_y.inverse_transform(test_y)
        # predict_y = scaler_y.inverse_transform(predict_y)

        plotfilename = 'seqLen:{0}_batchsize:{1}_epochs:{2}_preMeasure:{3}_dropout:{4}'.format(sequence_length,
                                                                                               batch_size, epochs,
                                                                                               predict_measure, dropout)
        title = plotfilename + '\nmse:{0}_mape:{1}'.format(mse, mape)
        plot_and_save(title, sequence_length, plotfilename, all_y, test_x, test_y, train_y, predict_y)

    fo.close()


if __name__ == '__main__':

    main()