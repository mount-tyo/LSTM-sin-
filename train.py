import numpy as np
import math
import time 
from matplotlib import pyplot as plt 
import os
import model as sinnn
from config import *

import torch as T
import torch.nn as nn
from torch import optim

# y=sin(x)のデータセットをN個分作成
def get_data_for_SIN_NN(N, Nte):  
    x = np.linspace(0, 2 * np.pi, N+Nte)
    # 学習データとテストデータに分ける
    ram = np.random.permutation(N+Nte)
    x_train = np.sort(x[ram[:N]])
    x_test = np.sort(x[ram[N:]])

    t_train = np.sin(x_train)
    t_test = np.sin(x_test)
    
    return x_train, t_train, x_test, t_test


def get_data_for_LSTM_NN(data_size, seq_length=50, freq=60, noise=0.00):
    '''Make sin() dataset for model 'LSTM_NN'.

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
    [[[0.0], [0.10452846326765346], [0.20791169081775931]], [[0.10452846326765346], [0.20791169081775931], [0.3090169943749474]]]
    >>> print(tl)
    [[0.3090169943749474], [0.40673664307580015]]
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


def mk_random_batch(train_x, train_t, batch_size):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。

    Args
    ----
    train_x     :   3 dimensions list
                    size : data_size * seq_length * vector_size
                    学習用データ. (e.g.) train_x = [ [[1.],[2.],[3.]], [[4.],[5.],[6.]] ]
    train_t     :   2 dimensions list
                    学習用ラベル. (e.g.) train_t = [ [4.], [7.] ]
    batch_size  :   int
                    バッチサイズ.

    Returns
    -------
    batch_x :   3 dimensions list
                size : batch_size * seq_length * vector_size
    batch_t :   2 dimension list
    torch.tensor(batch_x)   :   torch.tensor
                                tensor of batch_x
    torch.tensor(batch_t)   :   torch.tensor
                                tensor of batch_t

    Examples
    --------
    """
    batch_x = []
    batch_t = []

    for i in range(batch_size):
        idx = np.random.randint(0, len(train_x) - 1)
        batch_x.append(train_x[idx])
        # print('i={}'.format(i))
        # print(batch_x)
        batch_t.append(train_t[idx])
    
    return T.tensor(batch_x), T.tensor(batch_t)


def training(N, Nte, bs, n_epoch, h_units=[1,1], act='sig'):

    # データセットの取得
    # x_train, t_train, x_test, t_test = get_data_for_SIN_NN(N, Nte)    # for SIN_NN
    x_train, t_train = get_data_for_LSTM_NN(N)                          # for LSTM_NN
    x_test, t_test = get_data_for_LSTM_NN(Nte)                          # for LSTM_NN

    # モデル：SIN_NNを使用する場合
    # x_test_torch = T.from_numpy(x_test.astype(np.float32).reshape(x_test.shape[0],1))
    # t_test_torch = T.from_numpy(t_test.astype(np.float32).reshape(t_test.shape[0],1))
    # モデル：LSTM_NNを使用する場合
    x_test_torch = T.tensor(x_test)
    t_test_torch = T.tensor(t_test)

    # Setup model
    # model = sinnn.SIN_NN(h_units, act)            # use model 'SIN_NN'
    model = sinnn.LSTM_NN(1, hidden_size, 1)        # use model 'LSTM_NN'
    optimizer = optim.Adam(model.parameters())      # 最適化関数の設定
    MSE = nn.MSELoss()                              # 損失関数の設定

    # loss格納配列
    tr_loss = []        # training用
    te_loss = []        # test用

    # ディレクトリを作成
    # if os.path.exists(save_path + "{}/Pred_bs{}_h{}".format(act,bs,h_units[0])) == False:
        # os.makedirs(save_path + "{}/Pred_bs{}_h{}".format(act,bs,h_units[0]))
    if os.path.exists(save_path + "Pred_bs{}_hs{}".format(bs,hidden_size)) == False:
        os.makedirs(save_path + "Pred_bs{}_hs{}".format(bs,hidden_size))

    # 時間を測定
    start_time = time.time()
    print("START")

    # 学習回数分のループ
    for epoch in range(1, n_epoch + 1):
        # training
        '''SIN_NN
        model.train()
        perm = np.random.permutation(N)
        sum_loss = 0
        for i in range(0, N, bs):
            x_batch = x_train[perm[i:i + bs]]
            t_batch = t_train[perm[i:i + bs]]
            # print(t_batch)
            optimizer.zero_grad()
            y_batch, t_batch = model(x_batch, t_batch)
            
            dd

            loss = MSE(y_batch, t_batch)
            loss.backward()
            optimizer.step()
            sum_loss += loss.data * bs
        '''
        model.train()
        sum_loss = 0
        for i in range(int(N / bs)):
            optimizer.zero_grad()
            # print('train_x\n{}'.format(train_x))
            # print('train_t\n{}'.format(train_t))
            data, label = mk_random_batch(x_train, t_train, bs)
            # print('data\n{}'.format(data))
            # print('label\n{}'.format(label))

            output = model(data)
            # print(output)
            loss = MSE(output, label)
            loss.backward()
            optimizer.step()
            sum_loss += loss.data * bs

        # 学習誤差の平均を計算
        ave_loss = sum_loss / N
        tr_loss.append(ave_loss)

        # テスト誤差
        model.eval()
        # y_test_torch = model.forward(x_test_torch)
        y_test_torch = model(x_test_torch, None)
        loss = MSE(y_test_torch, t_test_torch)
        te_loss.append(loss.data)

        '''バッチ処理?
        model.eval()
        sum_loss = 0
        for i in range(int(Nte / bs)):
            offset = i * bs
            data, label = torch.tensor(x_test[offset:offset+bs]), torch.tensor(t_test[offset:offset+bs])
            output = model(data, None)
        '''
            

        # 学習過程を出力
        if epoch % 100 == 1:
            print("Ep/MaxEp     tr_loss     te_loss")

        if epoch % 10 == 0:
            print("{:4}/{}  {:10.5}   {:10.5}".format(epoch, n_epoch, ave_loss, float(loss.data)))

            # 誤差をリアルタイムにグラフ表示
            '''
            plt.plot(tr_loss, label = "training")
            plt.plot(te_loss, label = "test")
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.xlabel("epoch")
            plt.ylabel("loss (MSE)")
            plt.pause(0.01)  # このコードによりリアルタイムにグラフが表示されたように見える
            plt.clf()
            '''

        if epoch % 20 == 0:
            # epoch20ごとのテスト予測結果
            plt.figure(figsize=(5, 4))
            y_test = model.predict(x_test)
            # print('x_test')
            # print(type(x_test))
            # print('t_test')
            # print(t_test)
            
            # plt.plot(x_test, t_test, label = "target")
            xlim_index = [i for i in range(Nte)]
            plt.plot(xlim_index, t_test, label = "target")
            # plt.plot(x_test, y_test, label = "predict")
            plt.plot(xlim_index, y_test, label = "predict")
            # plt.legend()
            plt.legend(loc='upper right', borderaxespad=0, fontsize=10)
            plt.grid(True)
            # plt.xlim(0, 2 * np.pi)
            plt.xlim(0, Nte+20)
            plt.ylim(-1.4, 1.4)
            plt.xlabel("x")
            # plt.ylabel("y")
            plt.ylabel("sin(x)")
            # save file path
            plt.savefig(save_path + "Pred_bs{}_hs{}/ep{}.png".format(bs,hidden_size,epoch))
            plt.clf()
            plt.close()

    print("END")

    # 経過時間
    total_time = int(time.time() - start_time)
    print("Time : {} [s]".format(total_time))

    # 誤差のグラフ作成
    plt.figure(figsize=(5, 4))
    plt.plot(tr_loss, label = "training")
    plt.plot(te_loss, label = "test")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE)")
    # save file path
    plt.savefig(save_path + "loss_history_bs{}_hs{}.png".format(bs,hidden_size))
    plt.clf()
    plt.close()

    # 学習済みモデルの保存
    # save model path
    T.save(model, save_path + "bs{}_hs{}_model.pt".format(bs,hidden_size))