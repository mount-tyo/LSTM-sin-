from config import *
import train


if __name__ == "__main__":
    '''
    for bs in bss:
        for h_units in h_unitss:
            for act in acts:
                train.training(N, Nte, bs, n_epoch, h_units, act)
    '''
    train.training(N, Nte, 20, 1000, [11,11], 'sig')