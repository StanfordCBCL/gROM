import sys

sys.path.append("../graphs")
sys.path.append("../network")

import matplotlib.pyplot as plt

def plot_history(history_train, history_test, label, folder):
    plt.figure()
    ax = plt.axes()

    ax.plot(history_train[0], history_train[1], color = 'red', label='train')
    ax.plot(history_test[0], history_test[1], color = 'blue', label='test')
    ax.legend()
    ax.set_xlim((history_train[0][0],history_train[0][-1]))

    ax.set_xlabel('epoch')
    ax.set_ylabel(label)
    ax.set_yscale('log')

    plt.savefig(folder + '/' + label + '.png')