from pytorch_lightning import Trainer

from dataset import TestLoader_std_lightning
from model.model_lightning import PSMNet_lightning
from model.model_lightning_deep import PSMNetDeep_lightning
from model.model_lightning_deconv import PSMNetDeconv_lightning

from dataset import vaihingen_evaluation as ve
from dataset import vaihingen_info as info

import argparse

from argparse import Namespace

def main():
    parser = argparse.ArgumentParser(description='PSMNet lightning')
    parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
    parser.add_argument('--datalist', default='', help='input data image list')
    parser.add_argument('--info_datapath', default='', help='information path')
    parser.add_argument('--savepath', default='', help='save result directory')
    parser.add_argument('--subfolder', default='', help='save result with subfolder')
    parser.add_argument('--loadmodel', default=None, help='loading model')
    parser.add_argument('--loadcsv', default='', help='loading model csv')

    args = parser.parse_args()

    print('***[args]: ', args)

    all_left_img, all_right_img, all_left_disp = ve.load_vaihingen_evluation(args.datalist, args.savepath, args.subfolder)
    #print(all_left_img)
    normalize = info.load_vaihingen_info(args.info_datapath)
    
    model = PSMNet_lightning(args.maxdisp)

    # refer to https://github.com/Lightning-AI/lightning/issues/525
    """hparams = {
    "max_disp":192,
    }
    namespace = Namespace(**hparams)"""

    #https://github.com/Lightning-AI/lightning/issues/89
    # too old 2019
    #trained_model = model.load_from_metrics(weights_path=args.loadmodel, tags_csv=args.loadcsv)

    #trained_model = model.load_from_checkpoint(checkpoint_path=args.loadmodel,  hparams_file=args.loadcsv, map_location=None)

    trained_model = model.load_from_checkpoint(checkpoint_path=args.loadmodel, map_location=None)

    data = TestLoader_std_lightning.TestImageFloderStd_lightning(all_left_img, all_right_img, all_left_disp, normalize, test_batch_size=1)

    #print(trained_model)
    # use gdb debug python
    # logger should be false
    # refer to https://github.com/Lightning-AI/lightning/issues/5488
    trainer = Trainer(devices=1, accelerator="gpu", enable_progress_bar=False, logger=False)
    trainer.test(trained_model, data)

    print("inference over.")

if __name__ == "__main__":
    main()    