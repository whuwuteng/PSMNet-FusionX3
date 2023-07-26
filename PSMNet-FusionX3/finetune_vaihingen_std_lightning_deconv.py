from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import KITTILoader_std_lightning
#from model.model_lightning import PSMNet_lightning
from model.model_lightning_deconv import PSMNetDeconv_lightning

from dataset import vaihingen_collector_file as vse
from dataset import vaihingen_info as info

import argparse

def main():
    parser = argparse.ArgumentParser(description='PSMNet')
    
    parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
    parser.add_argument('--train_datapath', default=None,
                    help='training data path')
    parser.add_argument('--val_datapath', default=None,
                    help='validation data path')
    parser.add_argument('--info_datapath', default=None,
                    help='information path') 
    parser.add_argument('--savemodel', default=None,
                    help='information path')
    parser.add_argument('--loadmodel', default=None,
                    help='loading model')
    args = parser.parse_args()

    print('***[args]: ', args)

    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = vse.datacollectorall(
    args.train_datapath, args.val_datapath)

    normalize = info.load_vaihingen_info(args.info_datapath)
    
    model = PSMNetDeconv_lightning(args.maxdisp)

    data = KITTILoader_std_lightning.myImageFloderStd_lightning(all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp, normalize, train_batch_size = 2, val_batch_size = 1)

    # not used for all
    checkpoint_callback = ModelCheckpoint(dirpath=args.savemodel,  save_top_k = -1, every_n_epochs = 5, filename='{epoch:03d}')

    trainer = Trainer(devices=4, accelerator="gpu", strategy="ddp", resume_from_checkpoint=args.loadmodel, callbacks=[checkpoint_callback], enable_progress_bar=False,logger=False)

    # set train ckpt_path make no sense
    # because Trainer can resume
    # refer to https://github.com/Lightning-AI/lightning/issues/4392
    
    trainer.fit(model, data, ckpt_path=args.loadmodel)

    # refero to 
    #trainer.fit(model, data)

if __name__ == "__main__":
    main()