import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class LSTM_NN(nn.Module):
    '''
    def backward()は定義しなくても、pytorchがいい感じにやってくれるらしい
    (ベクトルの流れ)
        input (batch_size x sequence_size x 1)  
        --LSTM-->  
        output (batch_size x sequence_size x hidden_size)  
        --(一部を取り出す)-->
        outout[:, -1, :] (batch_size x 1 x hidden_size)  
        --Linear-->  
        output (batch_size x 1 x 1) 
    '''
    def __init__(self, inputDim, hiddenDim, outputDim):
        '''
        inputDim  : 実際は1. 各時刻の入力はsin値ただ1つ.
        outputDim : 実際は1. 各時刻の出力はsin値ただ1つ.
        '''
        super(LSTM_NN, self).__init__()
        self.rnn = nn.LSTM(input_size=inputDim,
                            hidden_size=hiddenDim,
                            batch_first=True)
        '''
        input_size  : 各時刻における入力ベクトルのサイズ
        hidden_size : LSTMの隠れ層ベクトルのサイズ
        batch_first : Trueなら、LSTMの入力テンソルの形式において、バッチサイズの次元が先頭に来る
        (本来のLSTMの入力テンソルの形式)
            Sequence_Length x Batch_Size x Vector_Size
        '''
        self.output_layer = nn.Linear(hiddenDim, outputDim)

    def forward(self, inputs, hidden0=None):
        '''
        inputs  : 時系列データ全体
        hidden0 : 隠れ層とセルの初期状態を意味するタプル
                  ex) hidden0 = (hid0, cell0)
        '''
        output, (hidden, cell) = self.rnn(inputs, hidden0) #LSTM層
        '''
        output : LSTMの各時刻の出力の系列
        hidden : 最後の隠れ層の状態
        cell   : 最後のセルの状態
        '''
        # pytorchのLSTMにhidden0をNoneで渡すと、ゼロベクトルとして扱うらしい
        output = self.output_layer(output[:, -1, :]) #全結合層
        '''
        output[:, -1, :]  :  時系列の最後の値(ベクトル)を参照している
                             output_layer()に入れることで、サイズ１のベクトルに変換される
        '''
        return output

    def predict(self, x):
        x = T.tensor(x)
        y = self.forward(x)
        # print(y)
        
        return y.data
    


# Neural Network
class SIN_NN(nn.Module):
    def __init__(self, h_units, act):
        super(SIN_NN, self).__init__()
        self.l1=nn.Linear(1, h_units[0])
        self.l2=nn.Linear(h_units[0], h_units[1])
        self.l3=nn.Linear(h_units[1], 1)

        if act == "relu":
            self.act = F.relu
        elif act == "sig":
            self.act = T.sigmoid

    def __call__(self, x, t):
        x = T.from_numpy(x.astype(np.float32).reshape(x.shape[0],1))
        t = T.from_numpy(t.astype(np.float32).reshape(t.shape[0],1))
        y = self.forward(x)
        return y, t

    def forward(self, x):
        h = self.act(self.l1(x))
        h = self.act(self.l2(h))
        h = self.l3(h)

        return h

    def predict(self, x):
        x = T.from_numpy(x.astype(np.float32).reshape(x.shape[0],1))
        y = self.forward(x)

        return y.data