from __future__ import print_function
import torch.utils.data as data
import os
import glob
import pdb

def load_vaihingen_data(folderlist, guide_folder):
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
    all_left_guide = []
    all_left_disp = []

    for current_file in filelist :
        filename = file_path + '/' + current_file[0: len(current_file)]
        #print('left image: ' + filename)
        all_left_img.append(filename)
        #index1_dir = current_file.find('/')
        #index2_dir = current_file.find('/', index1_dir + 1)
        index2_dir = current_file.rfind('/')
        index1_dir = current_file.rfind('/', 0, index2_dir)

        filename = file_path + '/' + current_file[0: index1_dir] + '/colored_1' + current_file[index2_dir: len(current_file)]
        #print('right image: ' + filename)
        all_right_img.append(filename)
        
        filename = file_path + '/' + current_file[0: index1_dir] + '/' + guide_folder + current_file[index2_dir: len(current_file)]
        #print('disp image: ' + filename)
        all_left_guide.append(filename)
        
        filename = file_path + '/' + current_file[0: index1_dir] + '/disp_occ' + current_file[index2_dir: len(current_file)]
        all_left_disp.append(filename)

    return all_left_img, all_right_img, all_left_guide, all_left_disp

def datacollectorall(train_filelist, val_filelist, guide_folder):

    left_train, right_train, guide_train_L, disp_train_L = load_vaihingen_data(train_filelist, guide_folder)
    left_val, right_val, guide_val_L, disp_val_L = load_vaihingen_data(val_filelist, guide_folder)

    return left_train, right_train, guide_train_L, disp_train_L, left_val, right_val, guide_val_L, disp_val_L

