import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np

class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])

        return output

def mkDataSet(data_size, data_length=5, freq=60., noise=0.00):
    """
    params\n
    data_size : データセットサイズ\n
    data_length : 各データの時系列長\n
    freq : 周波数\n
    noise : ノイズの振幅\n
    returns\n
    train_x : トレーニングデータ（t=1,2,...,size-1の値)\n
    train_t : トレーニングデータのラベル（t=sizeの値）\n
    """
    train_x = []
    train_t = []

    for offset in range(data_size):
        # noise ON
        # train_x.append([[math.sin(2 * math.pi * (offset + i) / freq) + np.random.normal(loc=0.0, scale=noise)] for i in range(data_length)])
        # noise OFF
        train_x.append([[math.sin(2 * math.pi * (offset + i) / freq)] for i in range(data_length)])
        train_t.append([math.sin(2 * math.pi * (offset + data_length) / freq)])

    return train_x, train_t

def mkRandomBatch(train_x, train_t, batch_size=10):
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
    
    return torch.tensor(batch_x), torch.tensor(batch_t)

def main():
    training_size = 4
    test_size = 4
    epochs_num = 1000
    hidden_size = 5
    batch_size = 2

    train_x, train_t = mkDataSet(training_size)
    test_x, test_t = mkDataSet(test_size)

    model = Predictor(1, hidden_size, 1)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()
            # print('train_x\n{}'.format(train_x))
            # print('train_t\n{}'.format(train_t))
            data, label = mkRandomBatch(train_x, train_t, batch_size)
            print('data\n{}'.format(data))
            # print('label\n{}'.format(label))

            output = model(data)
            print(output)
            dd
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            training_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)

        #test
        test_accuracy = 0.0
        for i in range(int(test_size / batch_size)):
            offset = i * batch_size
            data, label = torch.tensor(test_x[offset:offset+batch_size]), torch.tensor(test_t[offset:offset+batch_size])
            output = model(data, None)

            test_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)
        
        training_accuracy /= training_size
        test_accuracy /= test_size

        print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
            epoch + 1, running_loss, training_accuracy, test_accuracy))


if __name__ == '__main__':
    main()