'''
1.用许多电池数据训练出一个基本模型
2.针对某一个测试电池数据，使用其前百分之五十用于训练模型，后百分之五十用于测试预测。
也就是说，对于训练数据集，指电池的整个生命周期数据；
对于测试集，数据只包含电池的生命周期数据的一部分，如前百分之五十。

'''

import pandas as pd
import numpy as np

#读取多个电池生命周期数据，集合到一个csv文件中，用于训练
def main():
    filename = "../data/2017_06_30_cell"#0_data
    usecols = [0,3,4,5,6,7,8,9,10]
    data=np.zeros((9,))
    for i in range(0,31):
        filename_curr =filename+str(i)+"_data.csv"
        df = pd.read_csv(filename_curr,sep=',',usecols=usecols)
        data_curr=np.array(df).astype(float)
        print(data_curr.shape)
        data = np.vstack([data,data_curr])
        print(data_curr.shape)

    print(data.shape)
    data = data[1:,:]
    data = pd.DataFrame(columns=['cell','cycle','IR','Tavg','Tmin','Tmax','chargetime','capacity','label'],data=data)
    data.to_csv('cells_multi_input.csv',index=False)


if __name__ == '__main__':
    main()