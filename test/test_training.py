import sys
import os
sys.path.append(os.getcwd())
from network1d.training import training

if __name__ == "__main__":
    graphs_folder = 'graphs/'
    data_location = 'test/test_data/'
    training(False, 0, graphs_folder, data_location)