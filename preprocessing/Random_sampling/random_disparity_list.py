import argparse
import os
import glob
from PIL import Image
import imageio
import numpy as np
import random
import pdb

def load_vaihingen_data(folderlist):
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
        index2_dir = current_file.rfind('/')
        index1_dir = current_file.rfind('/', 0, index2_dir)

        filename = file_path + '/' + current_file[0: index1_dir] + '/disp_occ' + current_file[index2_dir: len(current_file)]
        #print('disp image: ' + filename)
        all_left_disp.append(filename)

    return all_left_disp

def RandomSampleDisp(disp_src, disp_tar, scale) :
    disp = Image.open(disp_src)
    d_width, d_height = disp.size
    
    disp = np.array(disp)
    mask = (disp > 0)

    valid_pixels = np.argwhere(mask)
    valid_total = len(valid_pixels)
    #print('total valid pixel: ' + str(valid_total))

    select_num = int(valid_total * scale)
    select_pixels = random.sample(list(valid_pixels), select_num)
    #print(select_pixels)

    sparse_img = np.zeros((d_height, d_width), dtype=np.uint16)
    sparse_img.fill(0)
    
    for pixel in select_pixels:
        sparse_img[pixel[0], pixel[1]] = disp[pixel[0], pixel[1]]

    # save image
    imageio.imwrite(disp_tar, sparse_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='show the histgram of disparity')

    parser.add_argument("--src", type=str, default='', help='input disparity image')
    parser.add_argument("--tar", type=str, default='', help='output disparity image')
    parser.add_argument("--txtlist", type=str, default='', help='input disparity image')
    parser.add_argument("--folder", type=str, default='', help='output disparity image')
    parser.add_argument('--scale', type=float ,default=0, help='random select scale')
    
    args = parser.parse_args()

    # test
    RandomSampleDisp(args.src, args.tar, args.scale)
    exit()

    filelist = load_vaihingen_data(args.txtlist)

    for current_file in filelist:
        index2_dir = current_file.rfind('/')
        index1_dir = current_file.rfind('/', 0, index2_dir)
        
        save_path = current_file[0: index1_dir] + '/' + args.folder 
        if not os.path.exists(save_path) :
            os.makedirs(save_path)
         
        filename = current_file[0: index1_dir] + '/' + args.folder + current_file[index2_dir: len(current_file)]

        #print("save: " + filename)

        RandomSampleDisp(current_file, filename, args.scale)

