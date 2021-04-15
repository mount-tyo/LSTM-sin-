import numpy as np
import math
def get_data_for_LSTM_NN(data_size, seq_length=50, freq=60, noise=0.00):
    '''Make sin dataset for model 'LSTM_NN'.

    Args
    ----
    data_size   :   int
                    シーケンスデータの総数
    seq_length  :   int
                    各シーケンスデータの長さ
    freq        :   int
                    sin波の周波数
    noise       :   float
                    sin波に付加するノイズの振幅

    Returns
    -------
    train_data  :   3 dimensions list
                    学習用データ.　sin()の値が格納.
    train_label :   2 dimensions list
                    学習用データに対するラベル.学習用データの次の値が格納.
    
    Examples
    --------
    >>> td, tl = get_data_for_LSTM_NN(3,2)
    >>> print(td)
    '''
    train_data = []
    train_label = []

    for offset in range(data_size):
        # noise ON
        # train_x.append([[math.sin(2 * math.pi * (offset + i) / freq) + np.random.normal(loc=0.0, scale=noise)] for i in range(data_length)])
        # noise OFF
        train_data.append([[math.sin(2 * math.pi * (offset + i) / freq)] for i in range(seq_length)])
        train_label.append([math.sin(2 * math.pi * (offset + seq_length) / freq)])
    
    return train_data, train_label

if __name__ == '__main__':
    td,tl=get_data_for_LSTM_NN(2,3)
    print(td)
    print('----')
    print(tl)