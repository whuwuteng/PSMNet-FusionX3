import argparse
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors
import imageio
import random
import pdb

def load_vaihingen_data(folderlist, src_folder):
    """load current file from the list"""
    file_path, filename = os.path.split(folderlist)

    filelist = []
    with open(folderlist) as w :
        content = w.readlines()
        content = [x.strip() for x in content] 
        for line in content :
            if line :
                filelist.append(line)

    all_left_disp = []

    for current_file in filelist :
        #index1_dir = current_file.find('/')
        #index2_dir = current_file.find('/', index1_dir + 1)
        index2_dir = current_file.rfind('/')
        index1_dir = current_file.rfind('/', 0, index2_dir)

        filename = file_path + '/' + current_file[0: index1_dir] + '/' + src_folder + current_file[index2_dir: len(current_file)]
        #print('disp image: ' + filename)
        all_left_disp.append(filename)

    return all_left_disp

def CreateRightDisp(disp_src, disp_tar, scale):
    disp = Image.open(disp_src)
    d_width, d_height = disp.size
    
    disp = np.array(disp)
    mask = (disp > 0)

    valid_pixels = np.argwhere(mask)

    right_disp = np.zeros((d_height, d_width), dtype=np.uint16)
    right_disp.fill(0) 
    for pixel in valid_pixels:
        disp_pixel = disp[pixel[0], pixel[1]]/scale
        #print(disp_pixel)
        rightx = int(pixel[1] - disp_pixel + 0.5)
        right_disp[pixel[0], rightx] = disp[pixel[0], pixel[1]]

    # save image
    imageio.imwrite(disp_tar, right_disp)

# for the ISPRS vaihingen
# it will be too sparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='show the histgram of disparity')

    parser.add_argument("--src", type=str, default='', help='input disparity image')
    parser.add_argument("--tar", type=str, default='', help='output disparity image')
    parser.add_argument("--txtlist", type=str, default='', help='input disparity image')
    parser.add_argument("--srcfolder", type=str, default='', help='input disparity image')
    parser.add_argument("--tarfolder", type=str, default='', help='output disparity image')
    parser.add_argument('--disp_scale', type=float ,default=256, help='random select scale')
    
    args = parser.parse_args()

    # test
    CreateRightDisp(args.src, args.tar, args.disp_scale)
    exit()

    filelist = load_vaihingen_data(args.txtlist, args.srcfolder)

    for current_file in filelist:
        index2_dir = current_file.rfind('/')
        index1_dir = current_file.rfind('/', 0, index2_dir)
        
        save_path = current_file[0: index1_dir] + '/' + args.tarfolder 
        if not os.path.exists(save_path) :
            os.makedirs(save_path)
         
        filename = current_file[0: index1_dir] + '/' + args.tarfolder + current_file[index2_dir: len(current_file)]

        #print(current_file)
        #print(filename)
        if not os.path.exists(filename):
            print(filename)
            CreateRightDisp(current_file, filename, args.disp_scale)


