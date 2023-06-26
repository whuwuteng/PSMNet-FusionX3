#! /bin/bash

python3 run.py --datapath /work/OT/ai4geo/users/tengw/KITTI_guide_stereo/2011_09_26_0011/ \
              --loadmodel /work/OT/ai4geo/users/tengw/KITTI_guide_stereo/result/finetune_24.tar \
              --output_dir /work/OT/ai4geo/users/tengw/KITTI_guide_stereo/result/ \
              --guided \
              --save \
              --verbose \
