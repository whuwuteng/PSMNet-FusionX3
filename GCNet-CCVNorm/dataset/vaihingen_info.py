
import os
import glob
import pdb

def load_vaihingen_info(info_path):
    normalize = None
    with open(info_path) as w :

        line = w.readline()
        mean_line = line.split()

        mean = []
        if len(mean_line) == 4 :
            mean.append(float(mean_line[1]))
            mean.append(float(mean_line[2]))
            mean.append(float(mean_line[3]))
        elif len(mean_line) == 2 :
            mean.append(float(mean_line[1]))

        line = w.readline()
        std_line = line.split()

        std = []
        if len(mean_line) == 4 :
            std.append(float(std_line[1]))
            std.append(float(std_line[2]))
            std.append(float(std_line[3]))
        elif len(mean_line) == 2 :
            std.append(float(std_line[1]))

        normalize = dict()
        normalize['mean'] = mean
        normalize['std'] = std

    return normalize