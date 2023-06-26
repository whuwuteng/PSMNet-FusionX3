#! /bin/bash

#python test.py --model_cfg misc/test_options.py --model_path /work/OT/ai4geo/users/tengw/KITTI_Depth_Completion/test/ckpt/\[ep-00\]giter-9000.ckpt --dataset kitti2015 --root_dir /work/OT/ai4geo/users/tengw/KITTI_Depth_Completion/kitti_stereo/data_scene_flow 


python test.py --model_cfg misc/test_options.py --model_path /work/OT/ai4geo/users/tengw/KITTI_Depth_Completion/test/ckpt/\[ep-00\]giter-9000.ckpt --dataset kitti2017 --rgb_dir /work/OT/ai4geo/users/tengw/KITTI_Depth_Completion/kitti2017/rgb --depth_dir /work/OT/ai4geo/users/tengw/KITTI_Depth_Completion/kitti2017/depth --save
