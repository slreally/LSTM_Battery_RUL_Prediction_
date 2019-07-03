import argparse
import os.path as osp
import datetime
import numpy as np
import lstm_model
import loss
from utils import plot_and_save,get_time
import load_data
import lstm_model

def main():

    parser = argparse.ArgumentParser(description='LSTM RUL Prediction')
    parser.add_argument('--filename', type=str, default="data/2017_06_30_cell40_data.csv")
    parser.add_argument('--output_path',type=str,default="snapshot/single_variable")
    parser.add_argument('--predict_measure', type=int, default=0, choices=[0,1])
    parser.add_argument('--sequence_length', type=int,default= 20)
    parser.add_argument('--split', default=0.5, help='split of train and test set')
    parser.add_argument('--batch_size', type=int, default= 32,
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default= 1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--dropout', default= 0.2)
    parser.add_argument('--saved_figure_path',default='result/single_variable/')
    parser.add_argument('--feature_num',type=int,default= 7,
                        help='single feature use 1,multi use 7')
    parser.add_argument('--usecols',default= [3, 4, 5, 6, 7, 8, 9, 10],
                        help='single feature imput use [9,10], multi use [3, 4, 5, 6, 7, 8, 9, 10]')
    parser.add_argument('--get_model_measure',default= 0,
                        help='0 for define model from begin, 1 for load exist model')

    args = parser.parse_args()
    split = args.split
    dropout_prob = args.dropout
    sequence_length = args.sequence_length
    batch_size = args.batch_size
    epochs = args.epochs
    predict_measure = args.predict_measure  # 0 for predicting one cycle,1 for predicting len(test(y)) cycles continuely, use current predicted value as the next input.
    filename = args.filename
    save_filepath = args.saved_figure_path
    feature_num = args.feature_num #用于训练的每个数据样本的特征数
    usecols=args.usecols #要读取的数据文件的列
    get_model_measure = args.get_model_measure

    loss_file_path = args.output_path
    loss_file_name = 'seqlen:{0}_mse_mape_{1}.txt'.format(str(sequence_length),str(get_time()))

    fo = open(osp.join(loss_file_path,loss_file_name),'w')

    fo.write(str('N,batch_size,epochs,mse,mape\n'))
    fo.flush()

    batch_size_list = [8,16,32,64,128]
    epochs_list = [50,75,100,150,200]

    dataloader = load_data.load_data(filename, sequence_length, split, usecols=usecols)
    train_x, train_y, test_x, test_y = dataloader.get_x_y()
    all_y = dataloader.get_all_y()

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], feature_num))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], feature_num))
    print(train_y.shape)

    lstm = lstm_model.lstm(sequence_length,feature_num,dropout_prob)
    model = get_model(lstm, get_model_measure)
    lstm.train_model(model,train_x,train_y,batch_size,epochs)
    predict_y = lstm.predict(model,test_x,pre_way=predict_measure)

    sca_x, sca_y = dataloader.get_scaler_x_y()
    train_y = sca_y.inverse_transform(train_y)
    test_y = sca_y.inverse_transform(test_y)
    predict_y = sca_y.inverse_transform(predict_y)

    mse = loss.get_rmse(test_y, predict_y)
    mape = loss.get_mape(test_y, predict_y)

    err_str = '{0},{1},{2},{3},{4}\n'.format(sequence_length, batch_size, epochs, mse, mape)
    fo.write(str(err_str))
    fo.flush()

    plotfilename = 'seqLen:{0}_batchsize:{1}_epochs:{2}_preMeasure:{3}_dropout:{4}'.format(sequence_length, batch_size,
                                                                                           epochs, predict_measure,
                                                                                           dropout_prob)
    title = plotfilename + '\nmse:{0}_mape:{1}'.format(mse, mape)
    plot_and_save(title,sequence_length,plotfilename,save_filepath,all_y,test_y,train_y,predict_y)
    fo.close()


def get_model(lstm, get_model_measure):
    if get_model_measure == 0: #define model by self.
        return lstm.build_model()
    elif get_model_measure ==1:#load model from file
        json_filepath = 'multi_battery/batch_size:32_epochs:50_premeas:0.json'
        model_weight_filepath= 'multi_battery/batch_size:32_epochs:50_premeas:0.h5'
        return lstm.load_model(json_filepath,model_weight_filepath)
    else:
        return None

if __name__ == '__main__':

    main()