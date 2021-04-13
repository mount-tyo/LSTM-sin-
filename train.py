import numpy as np 
import time 
from matplotlib import pyplot as plt 
import os
import model as sinnn
from config import *

import torch as T
import torch.nn as nn
from torch import optim

# y=sin(x)のデータセットをN個分作成
def get_data(N, Nte):
    x = np.linspace(0, 2 * np.pi, N+Nte)
    # 学習データとテストデータに分ける
    ram = np.random.permutation(N+Nte)
    x_train = np.sort(x[ram[:N]])
    x_test = np.sort(x[ram[N:]])

    t_train = np.sin(x_train)
    t_test = np.sin(x_test)

    return x_train, t_train, x_test, t_test


def training(N, Nte, bs, n_epoch, h_units, act):

    # データセットの取得
    x_train, t_train, x_test, t_test = get_data(N, Nte)
    x_test_torch = T.from_numpy(x_test.astype(np.float32).reshape(x_test.shape[0],1))
    t_test_torch = T.from_numpy(t_test.astype(np.float32).reshape(t_test.shape[0],1))

    # モデルセットアップ
    # model = sinnn.SIN_NN(h_units, act)
    model = sinnn.LSTM_NN(1, 11, 1)
    optimizer = optim.Adam(model.parameters())
    MSE = nn.MSELoss()

    # loss格納配列
    tr_loss = []
    te_loss = []

    # ディレクトリを作成
    if os.path.exists(save_path + "{}/Pred_bs{}_h{}".format(act,bs,h_units[0])) == False:
        os.makedirs(save_path + "{}/Pred_bs{}_h{}".format(act,bs,h_units[0]))

    # 時間を測定
    start_time = time.time()
    print("START")

    # 学習回数分のループ
    for epoch in range(1, n_epoch + 1):
        model.train()
        perm = np.random.permutation(N)
        sum_loss = 0
        for i in range(0, N, bs):
            x_batch = x_train[perm[i:i + bs]]
            t_batch = t_train[perm[i:i + bs]]

            optimizer.zero_grad()
            y_batch, t_batch = model(x_batch, t_batch)

            loss = MSE(y_batch, t_batch)
            loss.backward()
            optimizer.step()
            sum_loss += loss.data * bs

        # 学習誤差の平均を計算
        ave_loss = sum_loss / N
        tr_loss.append(ave_loss)

        # テスト誤差
        model.eval()
        y_test_torch = model.forward(x_test_torch)
        loss = MSE(y_test_torch, t_test_torch)
        te_loss.append(loss.data)

        # 学習過程を出力
        if epoch % 100 == 1:
            print("Ep/MaxEp     tr_loss     te_loss")

        if epoch % 10 == 0:
            print("{:4}/{}  {:10.5}   {:10.5}".format(epoch, n_epoch, ave_loss, float(loss.data)))

            # 誤差をリアルタイムにグラフ表示
            plt.plot(tr_loss, label = "training")
            plt.plot(te_loss, label = "test")
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.xlabel("epoch")
            plt.ylabel("loss (MSE)")
            plt.pause(0.1)  # このコードによりリアルタイムにグラフが表示されたように見える
            plt.clf()

        if epoch % 20 == 0:
            # epoch20ごとのテスト予測結果
            plt.figure(figsize=(5, 4))
            y_test = model.predict(x_test)
            plt.plot(x_test, t_test, label = "target")
            plt.plot(x_test, y_test, label = "predict")
            plt.legend()
            plt.grid(True)
            plt.xlim(0, 2 * np.pi)
            plt.ylim(-1.2, 1.2)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(save_path + "{}/Pred_bs{}_h{}/ep{}.png".format(act,bs,h_units[0],epoch))
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
    plt.savefig(save_path + "{}/loss_history_bs{}_h{}.png".format(act,bs,h_units[0]))
    plt.clf()
    plt.close()

    # 学習済みモデルの保存
    T.save(model, save_path + "{}/{}_bs{}_h{}_model.pt".format(act,act,bs,h_units[0]))