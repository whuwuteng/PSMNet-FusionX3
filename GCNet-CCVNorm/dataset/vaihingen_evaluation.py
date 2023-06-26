from __future__ import print_function
import torch.utils.data as data
import os
import glob
import pdb

def load_vaihingen_evluation(folderlist, savedisp, subfolder):
    """load current file from the list"""
    file_path, filename = os.path.split(folderlist)

    filelist = []
    with open(folderlist) as w :
        content = w.readlines()
        content = [x.strip() for x in content] 
        for line in content :
            if line :
                filelist.append(line)

    all_left_img = []
    all_right_img = []
    all_left_disp = []

    for current_file in filelist :
        filename = file_path + '/' + current_file[0: len(current_file)]
        #print('left image: ' + filename)
        all_left_img.append(filename)
        index1_dir = current_file.find('/')
        index2_dir = current_file.find('/', index1_dir + 1)
        filename = file_path + '/' + current_file[0: index1_dir] + '/colored_1' + current_file[index2_dir: len(current_file)]
        #print('right image: ' + filename)
        all_right_img.append(filename)
        filename = savedisp + '/' + current_file[0: index1_dir] + '/' + subfolder + current_file[index2_dir: len(current_file)]
        #print('disp image: ' + filename)
        all_left_disp.append(filename)

    return all_left_img, all_right_img, all_left_disp


def load_vaihingen_evluationEx(folderlist, savedisp, subfolder):
    """load current file from the list"""
    file_path, filename = os.path.split(folderlist)

    filelist = []
    with open(folderlist) as w :
        content = w.readlines()
        content = [x.strip() for x in content] 
        for line in content :
            if line :
                filelist.append(line)

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    all_bh_ratio = []

    for current_file in filelist :
        filename = file_path + '/' + current_file[0: len(current_file)]
        #print('left image: ' + filename)
        all_left_img.append(filename)
        index1_dir = current_file.find('/')
        index2_dir = current_file.find('/', index1_dir + 1)
        filename = file_path + '/' + current_file[0: index1_dir] + '/colored_1' + current_file[index2_dir: len(current_file)]
        #print('right image: ' + filename)
        all_right_img.append(filename)
        filename = savedisp + '/' + current_file[0: index1_dir] + '/' + subfolder + current_file[index2_dir: len(current_file)]
        #print('disp image: ' + filename)
        all_left_disp.append(filename)
        filename = file_path + '/' + current_file[0: index1_dir] + '/BHratio.txt'
        all_bh_ratio.append(filename)

    return all_left_img, all_right_img, all_left_disp, all_bh_ratio