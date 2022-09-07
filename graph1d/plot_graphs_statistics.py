from ast import Num
import sys
import os
sys.path.append(os.getcwd())
import tools.io_utils as io
import matplotlib 
import numpy as np
import scipy
from scipy import interpolate
import dgl
import torch as th
from tqdm import tqdm
import json
import random
import pathlib
import tools.plot_tools as pt
import matplotlib.pyplot as plt
import generate_graphs as gg
import generate_normalized_graphs as gng
from mpl_toolkits import mplot3d
import math

def number_format(value):
    exponent = np.floor(math.log(np.abs(value),10))
    base = value / 10**exponent
    if base > 9.99:
        base = 1
        exponent = exponent + 1

    # if exponent != 0:
    #     return '${:.2f} \cdot 10^{{{:.0f}}}$'.format(base, exponent)
    # else:
    #     return '${:.2f}$'.format(base, exponent)
    base_sign = '$+$'
    if base < 0:
        base_sign = '$-$'
    exponent_sign = '$+$'
    if exponent < 0:
        exponent_sign = '$-$'
    return '{:}{:.2f}E{:}{:02d}'.format(base_sign, np.abs(base), 
                                        exponent_sign, int(np.abs(exponent)))

def write_table(graphs):
    statistics = {}
    fields = {'node': ['area', 'pressure', 
                       'flowrate', 'dt'], 
              'edge': ['distance']}
    statistics = gng.compute_statistics(graphs, fields, statistics)
    print(statistics)
    # pressure flowrate area d deltat
    print('\\begin{table}[htbp]')
    print('\\centering')
    print('\\begin{tabular} {c c c c c c}')
    print('& pressure [mmHg] & flowrate [cc/s] & area [cm$^2$] & $d$ [cm] & $\Delta t$ [s] \\\\') 
    print('\\toprule')
    print('min & {:} & {:} & {:} & {:} & {:} \\\\'.format(
                                   number_format(statistics['pressure']['min']),
                                   number_format(statistics['flowrate']['min']),
                                   number_format(statistics['area']['min']),
                                   number_format(statistics['distance']['min']),
                                   number_format(statistics['dt']['min'])))
    print('max & {:} & {:} & {:} & {:} & {:} \\\\'.format(
                                   number_format(statistics['pressure']['max']),
                                   number_format(statistics['flowrate']['max']),
                                   number_format(statistics['area']['max']),
                                   number_format(statistics['distance']['max']),
                                   number_format(statistics['dt']['max'])))
    print('mean & {:} & {:} & {:} & {:} & {:} \\\\'.format(
                                number_format(statistics['pressure']['mean']),
                                number_format(statistics['flowrate']['mean']),
                                number_format(statistics['area']['mean']),
                                number_format(statistics['distance']['mean']),
                                number_format(statistics['dt']['mean'])))
    print('st.~dev. & {:} & {:} & {:} & {:} & {:} \\\\'.format(
                            number_format(statistics['pressure']['stdv']),
                            number_format(statistics['flowrate']['stdv']),
                            number_format(statistics['area']['stdv']),
                            number_format(statistics['distance']['stdv']),
                            number_format(statistics['dt']['stdv'])))

    print('\\bottomrule')
    print('\\end{tabular}')
    print('\\end{table}')


"""
This function makes boxplots out of statistics on the set of graphs
"""
if __name__ == "__main__":
    data_location = io.data_location()
    input_dir = data_location + 'graphs/'

    resample = 20
    data_pressure = []
    data_flowrate = []
    graph_names = []
    graphs = gng.load_all_graphs(input_dir)
    for graph_name in graphs:
        graph_names.append(graph_name.replace(".grph",""))
        graph = graphs[graph_name]
        pressure = graph.ndata['pressure'].detach().numpy().flatten()
        data_pressure.append(pressure[::resample])
        flowrate = graph.ndata['flowrate'].detach().numpy().flatten()
        data_flowrate.append(flowrate[::resample])

    fig = plt.figure(figsize =(10, 7))

    plot_fliers = True
    # pressure
    ax = fig.add_subplot(211)
    ax.set_xticklabels([])
    ax.set_ylabel('Pressure [mmHg]')
    plt.xticks(rotation = 90, fontsize=7, color = "white")
    flierprops = dict(marker='o', markerfacecolor='#54A0C0', markersize=2,
                  linestyle='none', markeredgecolor='#54A0C0')
    plt.boxplot(data_pressure, patch_artist=True, flierprops=flierprops,
                showfliers = plot_fliers)

    # flowrate
    ax = fig.add_subplot(212)
    ax.set_xticklabels([])
    ax.set_ylabel('Flowrate [cc/s]')
    ax.set_xlabel('Models in dataset')
    plt.xticks(rotation = 90, fontsize=7, color = "white")
    flierprops = dict(marker='o', markerfacecolor='#54A0C0', markersize=2,
                linestyle='none', markeredgecolor='#54A0C0')
    plt.boxplot(data_flowrate, patch_artist=True,  flierprops=flierprops,
                showfliers = plot_fliers)

    plt.savefig('graphs_statistics.eps', format='eps')

    write_table(graphs)




        



