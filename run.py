import argparse
import os.path as osp
import datetime
import numpy as np
import lstm_model
import loss
from utils import plot_and_save,get_time
import load_data

def main():

    parser = argparse.ArgumentParser(description='LSTM RUL Prediction')
    parser.add_argument('--filename', type=str, default="data/2017_06_30_cell0_data.csv")
    parser.add_argument('--output_path',type=str,default="snapshot/single_variable")
    parser.add_argument('--predict_measure', type=int, default=0, choices=[0,1])
    parser.add_argument('--sequence_length', type=int,default=54)
    parser.add_argument('--split', default=0.5, help='split of train and test set')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
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

    loss_file_path = argparse.output_path
    loss_file_name = 'seqlen:{0}_mse_mape_{1}.txt'.format(str(sequence_length),str(get_time()))

    fo = open(osp.join(loss_file_path,loss_file_name),'w')

    fo.write(str('N,batch_size,epochs,mse,mape\n'))
    fo.flush()

    batch_size_list = [8,16,32,64,128]
    epochs_list = [50,75,100,150,200]

    dataloader = load_data.load_data(filename + ".csv", sequence_length, split, usecols=[9, 10])
    train_x, train_y, test_x, test_y = dataloader.get_x_y()
    all_y = dataloader.get_all_y()

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    print(train_y.shape)
    predict_y = train_model(train_x, train_y, test_x,test_y,batch_size, epochs, predict_measure, dropout)

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
                                                                                           dropout)
    title = plotfilename + '\nmse:{0}_mape:{1}'.format(mse, mape)
    plot_and_save(title, sequence_length, plotfilename, all_y, test_x, test_y, train_y, predict_y)

    fo.close()


if __name__ == '__main__':

    main()