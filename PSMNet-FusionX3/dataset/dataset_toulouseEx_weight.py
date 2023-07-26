"""
Read toulouse data structure
Note:
- RGB normalize
- fix size random crop
"""

import os
import random
import numpy as np
from easydict import EasyDict
from skimage import io
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


# for toulouse dataset
# add the weight in the training 
class DatasetToulouseWeight(Dataset):
    FIXED_SHAPE = (1024,1024) # NOTE: Crop top; different from Toulouse2020
    #FIXED_SHAPE = (256, 1216) # NOTE: Crop top; different from dataset_kitti2017
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

    def __init__(self, datalist, output_size, mode, guideL_folder, guideR_folder, normalize=__imagenet_stats, disp_scale=256, fix_random_seed=False):
        # Check arguments
        self.mode = mode
        self.disp_scale = disp_scale
        self.output_size = output_size
        self.guideL_folder = guideL_folder
        self.guideR_folder = guideR_folder
        if fix_random_seed:
            random.seed(100)
            np.random.seed(seed=100)

        # Get all data path
        self.left_data_path, self.right_data_path = get_train_datapath(datalist, guideL_folder, guideR_folder)

        # Define data transform
        self.transform = EasyDict()
        if self.mode in ['train']:
            self.transform.rgb = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(**normalize)
            ])
            self.transform.depth = transforms.Compose([
                transforms.ToPILImage(mode='F'), # NOTE: is this correct?!
                transforms.ToTensor()
            ])
        else: # val
            self.transform.rgb = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(**normalize)
            ])
            self.transform.depth = transforms.Compose([
                transforms.ToPILImage(mode='F'), # NOTE: is this correct?!
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        # Get data
        while(True):
            try: # NOTE: there are broken images in the dataset; skip those broken images
                left_rgb = read_rgb(self.left_data_path['rgb'][idx])
                img_h, img_w = left_rgb.shape[:2]
                right_rgb = read_rgb(self.right_data_path['rgb'][idx])
            except: # Encounter broken RGB
                idx = random.randint(0, len(self.left_data_path['rgb']))
                continue
            left_sdepth = read_depth(self.left_data_path['sdepth'][idx], self.disp_scale)
            left_weight = read_cmp(self.left_data_path['cmp'][idx], 255.0)
            left_depth = read_depth(self.left_data_path['depth'][idx], self.disp_scale)
            right_sdepth = read_depth(self.right_data_path['sdepth'][idx], self.disp_scale)
            right_weight = read_cmp(self.right_data_path['cmp'][idx], 255.0)
            right_depth = read_depth(self.right_data_path['depth'][idx], self.disp_scale)
            break

        # Crop to fixed size
        def crop_fn(x):
            start_h = img_h - self.FIXED_SHAPE[0]
            start_w = 0
            return x[start_h:start_h+self.FIXED_SHAPE[0], start_w:start_w+self.FIXED_SHAPE[1]]
        left_rgb, left_sdepth, left_weight, left_depth = list(map(crop_fn, [left_rgb, left_sdepth, left_weight, left_depth]))
        right_rgb, right_sdepth, right_weight, right_depth = list(map(crop_fn, [right_rgb, right_sdepth, right_weight, right_depth]))
        if self.output_size[0] < self.FIXED_SHAPE[0] or self.output_size[1] < self.FIXED_SHAPE[1]:
            x1 = random.randint(0, self.FIXED_SHAPE[1]-self.output_size[1])
            y1 = random.randint(0, self.FIXED_SHAPE[0]-self.output_size[0])
            def rand_crop(x):
                return x[y1:y1+self.output_size[0], x1:x1+self.output_size[1]]
            left_rgb, left_sdepth, left_weight, left_depth = list(map(rand_crop, [left_rgb, left_sdepth, left_weight, left_depth]))
            right_rgb, right_sdepth, right_weight, right_depth = list(map(rand_crop, [right_rgb, right_sdepth, right_weight, right_depth]))
        # Perform transforms
        data = dict()
        data['left_rgb'], data['right_rgb'] = list(map(self.transform.rgb, [left_rgb, right_rgb]))
        data['left_sd'], data['right_sd'] = list(map(self.transform.depth, [left_sdepth, right_sdepth]))
        data['left_cmp'], data['right_cmp'] = list(map(self.transform.depth, [left_weight, right_weight]))
        data['left_d'], data['right_d'] = list(map(self.transform.depth, [left_depth, right_depth]))
        #data['left_d'], data['right_d'] = list([left_depth, right_depth])
        data['width'] = img_w

        return data

    def __len__(self):
        return len(self.left_data_path['rgb'])

def read_rgb(path):
    """ Read raw RGB and DO NOT perform any process to the image """
    rgb = io.imread(path)
    return rgb


def read_depth(path, disp_scale):
    """ Depth maps (annotated and raw Velodyne scans) are saved as uint16 PNG images,
        which can be opened with either MATLAB, libpng++ or the latest version of
        Python's pillow (from PIL import Image). A 0 value indicates an invalid pixel
        (ie, no ground truth exists, or the estimation algorithm didn't produce an
        estimate for that pixel). Otherwise, the depth for a pixel can be computed
        in meters by converting the uint16 value to float and dividing it by 256.0:

        disp(u,v)  = ((float)I(u,v))/256.0;
        valid(u,v) = I(u,v)>0;
    """
    depth = Image.open(path)
    depth = np.array(depth).astype(np.float32) / disp_scale
    return depth[:, :, np.newaxis]

def read_cmp(path, disp_scale):
    depth = Image.open(path)
    depth = np.array(depth).astype(np.float32) / disp_scale
    depth = depth * 5.0
    depth[depth == 0] = 1.0
    return depth[:, :, np.newaxis]



def get_train_datapath(folderlist, guideL, guideR):
    """ Read path to all data """
    """load current file from the list"""
    left_data_path = {'rgb': [], 'sdepth': [], 'cmp': [],'depth': []}
    right_data_path = {'rgb': [], 'sdepth': [], 'cmp': [], 'depth': []}
    
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
    all_right_guide = []
    all_left_weight = []
    all_right_weight = []
    all_left_disp = []
    all_right_disp = []

    for current_file in filelist :
        filename = file_path + '/' + current_file[0: len(current_file)]
        #print('left image: ' + filename)
        if not os.path.exists(filename) :
            print(filename + ' not exists')
        all_left_img.append(filename)
        #index1_dir = current_file.find('/')
        #index2_dir = current_file.find('/', index1_dir + 1)
        index2_dir = current_file.rfind('/')
        index1_dir = current_file.rfind('/', 0, index2_dir)

        filename = file_path + '/' + current_file[0: index1_dir] + '/colored_1' + current_file[index2_dir: len(current_file)]
        #print('right image: ' + filename)
        all_right_img.append(filename)
        
        #filename = file_path + '/' + current_file[0: index1_dir] + '/guide0_01' + current_file[index2_dir: len(current_file)]
        filename = file_path + '/' + current_file[0: index1_dir] + '/' + guideL + current_file[index2_dir: len(current_file)]
        #print('disp image: ' + filename)
        if not os.path.exists(filename) :
            print(filename + ' not exists')
        all_left_guide.append(filename)

        file_main, file_extension = os.path.splitext(filename)
        filename = file_main + "_uncertainty.png"
        if not os.path.exists(filename) :
            print(filename + ' not exists')
        all_left_weight.append(filename)

        #filename = file_path + '/' + current_file[0: index1_dir] + '/guideR0_01' + current_file[index2_dir: len(current_file)]
        filename = file_path + '/' + current_file[0: index1_dir] + '/' + guideR + current_file[index2_dir: len(current_file)]
        #print('disp image: ' + filename)
        if not os.path.exists(filename) :
            print(filename + ' not exists')
        all_right_guide.append(filename)

        file_main, file_extension = os.path.splitext(filename)
        filename = file_main + "_uncertainty.png"
        if not os.path.exists(filename) :
            print(filename + ' not exists')
        all_right_weight.append(filename)

        filename = file_path + '/' + current_file[0: index1_dir] + '/disp_occ' + current_file[index2_dir: len(current_file)]
        if not os.path.exists(filename) :
            print(filename + ' not exists')
        all_left_disp.append(filename)
        
        filename = file_path + '/' + current_file[0: index1_dir] + '/disp_occR' + current_file[index2_dir: len(current_file)]
        if not os.path.exists(filename) :
            print(filename + ' not exists')
        all_right_disp.append(filename)

    for count, file_left in enumerate(all_left_img):
        # Add to list
        left_data_path['rgb'].append(file_left)
        left_data_path['sdepth'].append(all_left_guide[count])
        left_data_path['cmp'].append(all_left_weight[count])
        left_data_path['depth'].append(all_left_disp[count])
        right_data_path['rgb'].append(all_right_img[count])
        right_data_path['sdepth'].append(all_right_guide[count])
        right_data_path['cmp'].append(all_right_weight[count])
        right_data_path['depth'].append(all_right_disp[count])

    return left_data_path, right_data_path

