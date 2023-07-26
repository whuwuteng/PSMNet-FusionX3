from pytorch_lightning import Trainer
import torch

from dataset.TestLoader_LiDAR_lightning import TestDatasetToulouse_lightning
from model.psmnet_lidar_lightning import PSMNetLiDAR_lightning

from model.psmnet_lidar_weight2x_lightning import PSMNetLiDARWeight2x_lightning

from dataset import vaihingen_evaluation_ex as ve
from dataset import vaihingen_info as info

import argparse

from argparse import Namespace

def main():
    parser = argparse.ArgumentParser(description='PSMNet lightning')
    parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
    parser.add_argument('--datalist', default='', help='input data image list')
    parser.add_argument('--info_datapath', default='', help='information path')
    parser.add_argument('--guideL', default='guide0_05', help='information path')
    parser.add_argument('--guideR', default='guideR0_05', help='information path')
    parser.add_argument('--savepath', default='', help='save result directory')
    parser.add_argument('--subfolder', default='', help='save result with subfolder')
    parser.add_argument('--loadmodel', default=None, help='loading model')

    args = parser.parse_args()

    print('***[args]: ', args)

    all_left_img, all_right_img, all_left_guide, all_right_guide, all_left_disp = ve.load_vaihingen_evluation(args.datalist, args.guideL, args.guideR, args.savepath, args.subfolder)

    #print(all_left_img)
    normalize = info.load_vaihingen_info(args.info_datapath)
    
    norm_mode = ['naive_categorical', # Applying categorical CBN on 3D-CNN in stereo matching network
                'naive_continuous', # Applying continuous CBN on 3D-CNN in stereo matching network
                'categorical', # Applying categorical CCVNorm on 3D-CNN in stereo matching network
                'continuous', # Applying continuous CCVNorm on 3D-CNN in stereo matching network
                'categorical_hier', # Applying categorical HierCCVNorm on 3D-CNN in stereo matching network
                ][4]

    model = PSMNetLiDARWeight2x_lightning(args.maxdisp, norm_mode = norm_mode)
    #model = PSMNetLiDARWeight_lightning(args.maxdisp, norm_mode = norm_mode)

    # refer to https://github.com/Lightning-AI/lightning/issues/525
    """hparams = {
    "max_disp":192,
    }
    namespace = Namespace(**hparams)"""

    #https://github.com/Lightning-AI/lightning/issues/89
    # too old 2019
    #trained_model = model.load_from_metrics(weights_path=args.loadmodel, tags_csv=args.loadcsv)

    # refer to https://github.com/Lightning-AI/lightning/issues/3629
    # refer to https://github.com/Lightning-AI/lightning/issues/3302
    #trained_model = model.load_from_checkpoint(checkpoint_path=args.loadmodel,  hparams_file=args.loadcsv, map_location=None)

    # can not test
    #trained_model = model.model.load_state_dict(torch.load(args.loadmodel, map_location='cuda:0'), strict=False)

    # not work
    #trained_model = model.load_from_checkpoint(checkpoint_path=args.loadmodel, map_location=None)

    data = TestDatasetToulouse_lightning(all_left_img, all_right_img, all_left_guide, all_right_guide, all_left_disp, normalize, test_batch_size=1)

    #print(trained_model)
    # use gdb debug python
    # logger should be false
    # refer to https://github.com/Lightning-AI/lightning/issues/5488

    # resume from the train
    trainer = Trainer(devices=1, accelerator="gpu", resume_from_checkpoint=args.loadmodel, enable_progress_bar=False, logger=False)
    
    # important, if not, the result is wrong
    # refer to https://github.com/Lightning-AI/lightning/issues/4392
    trainer.test(model, data, ckpt_path=args.loadmodel)

    print("inference over.")

if __name__ == "__main__":
    main()    