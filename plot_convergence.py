import tools.plot_tools as pt
import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    tr = np.zeros(4)
    te = np.zeros(4)

    with open('models/24.08.2022_15.06.49/history.bnr', 'rb') as f:
        history = pickle.load(f)
    
    tr[0] = history['train_rollout'][1][-1]
    te[0] = history['test_rollout'][1][-1]
    
    with open('models/24.08.2022_16.11.58/history.bnr', 'rb') as f:
        history = pickle.load(f)
    
    tr[1] = history['train_rollout'][1][-1]
    te[1] = history['test_rollout'][1][-1]

    with open('models/24.08.2022_18.30.44/history.bnr', 'rb') as f:
        history = pickle.load(f)
    
    tr[2] = history['train_rollout'][1][-1]
    te[2] = history['test_rollout'][1][-1]
    
    with open('models/24.08.2022_09.59.44/history.bnr', 'rb') as f:
        history = pickle.load(f)
    
    tr[3] = history['train_rollout'][1][-1]
    te[3] = history['test_rollout'][1][-1]

    sizes = [20, 40, 60, 95]

    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.set_aspect('auto')
    ax.plot(sizes, tr, '-d', linewidth = 3, label='train')
    ax.plot(sizes, te, '-d', linewidth = 3, label='test')
    ax.legend()
    ax.set_xlim((20,95))

    ax.set_xlabel('Dataset size')
    ax.set_ylabel('Rollut error')
    plt.tight_layout()

    plt.legend(frameon=False)   
    plt.savefig('convergence.eps')
