from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.dataset_toulouseEx_lightning import DatasetToulouse_lightning
from model.psmnet_lidar_lightning import PSMNetLiDAR_lightning


from dataset import vaihingen_collector_file as vse
from dataset import vaihingen_info as info

import argparse

# too much memory
def main():
    parser = argparse.ArgumentParser(description='PSMNet LiDAR')
    
    parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
    parser.add_argument('--train_datapath', default=None,
                    help='training data path')
    parser.add_argument('--val_datapath', default=None,
                    help='validation data path')
    parser.add_argument('--info_datapath', default=None,
                    help='information path')
    parser.add_argument('--guideL', default='guide0_05',
                    help='information path')
    parser.add_argument('--guideR', default='guideR0_05',
                    help='information path')
    parser.add_argument('--savemodel', default=None,
                    help='information path') 
    parser.add_argument('--loadmodel', default=None,
                    help='loading model')


    args = parser.parse_args()

    print('***[args]: ', args)

    normalize = info.load_vaihingen_info(args.info_datapath)
    
    norm_mode = ['naive_categorical', # Applying categorical CBN on 3D-CNN in stereo matching network
                'naive_continuous', # Applying continuous CBN on 3D-CNN in stereo matching network
                'categorical', # Applying categorical CCVNorm on 3D-CNN in stereo matching network
                'continuous', # Applying continuous CCVNorm on 3D-CNN in stereo matching network
                'categorical_hier', # Applying categorical HierCCVNorm on 3D-CNN in stereo matching network
                ][4]

    # categorical need too much momery
    model = PSMNetLiDAR_lightning(args.maxdisp, norm_mode = norm_mode)

    train_output_size = (256, 512)
    val_output_size =  (256, 512) # NOTE: set to (256, 1216) if there is enough gpu memory

    data = DatasetToulouse_lightning(args.train_datapath, train_output_size, args.val_datapath, val_output_size, args.guideL, args.guideR, normalize, train_batch_size = 3, val_batch_size = 1)

    checkpoint_callback = ModelCheckpoint(dirpath=args.savemodel, save_top_k = -1, every_n_epochs = 5, filename='{epoch:03d}')

    trainer = Trainer(devices=4, accelerator="gpu", strategy="ddp", resume_from_checkpoint=args.loadmodel,callbacks=[checkpoint_callback], enable_progress_bar=False,logger=False)

    trainer.fit(model, data, ckpt_path=args.loadmodel)

if __name__ == "__main__":
    main()