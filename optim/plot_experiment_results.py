import matplotlib
import csv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    filename = '/Users/luca/Downloads/export.csv'

    var1 = []
    var2 = []
    var1_name = 'loss'
    var2_name = 'global_error'
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        count = 0
        for row in spamreader:
            rowlist = row[0].replace('"','').split(',')
            if count == 0:
                index1 = rowlist.index(var1_name)
                index2 = rowlist.index(var2_name)
            else:
                if rowlist[index1] != '' and rowlist[index2] != '':
                    var1.append(float(rowlist[index1]))
                    var2.append(float(rowlist[index2]))
            count = count + 1

    fig, ax = plt.subplots(1)

    ax.scatter(var1, var2)

    xmin = 0
    xmax = 30
    ymin = 0
    ymax = .2
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    ax.set_xlabel('Loss')
    ax.set_ylabel('Global error')
    plt.show()
