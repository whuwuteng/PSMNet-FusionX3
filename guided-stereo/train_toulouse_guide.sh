#! /bin/bash

python train_toulouse.py --train_datapath /work/OT/ai4geo/users/tengw/stereobenchmark_AI4GEO/Toulouse_AI4GEO2020-stereo_urban/training/toulouse_ai4geo_trainlist_guide.txt \
              --val_datapath /work/OT/ai4geo/users/tengw/stereobenchmark_AI4GEO/Toulouse_AI4GEO2020-stereo_urban/training/toulouse_ai4geo_vallist_guide.txt \
              --savemodel /work/OT/ai4geo/users/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo0_025/ \
              --loadmodel /work/OT/ai4geo/users/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo0_025/finetune_374.tar \
              --resume \
              --folder guide0_025 \
              --epoch_start 375 \
              --guided \
              --epochs 800
