#!/bin/bash

# run a folder example
BASEDIR=$(pwd -L)
echo "$BASEDIR"

#modelpath="/work/OT/ai4geo/users/tengw/Toulouse_AI4GEO2020_Model_echo/PSMnet_LiDAR/finetune_364.tar"

#modelpath="/work/OT/ai4geo/users/tengw/DublinCity_Model_echo/PSMnet_LiDAR/finetune_324.tar"

#modelpath="/work/OT/ai4geo/users/tengw/Toulouse_AI4GEO2020_Model_echo/PSMnet_LiDAR_norm/finetune_359.tar"

#modelpath="/work/OT/ai4geo/users/tengw/Toulouse_AI4GEO2020_Model_echo/PSMnet_LiDAR_deux/finetune_359.tar"

#modelpath="/work/OT/ai4geo/users/tengw/Toulouse_AI4GEO2020_Model_echo/PSMnet_LiDAR_guide/finetune_519.tar"

#modelpath="/work/OT/ai4geo/users/tengw/Toulouse_AI4GEO2020_Model_echo/PSMnet_LiDAR_full/finetune_439.tar"

#modelpath="/work/OT/ai4geo/users/tengw/Toulouse_AI4GEO2020_Model_echo/PSMnet_LiDAR_basic2x/finetune_44.tar"

modelpath="/work/OT/ai4geo/users/tengw/Toulouse_AI4GEO2020_Model_echo/PSMnet_LiDAR_deux2x/finetune_34.tar"

Method="PSMnet_LiDAR_deux2x"

data_path="/work/OT/ai4geo/users/tengw/stereobenchmark_AI4GEO/Toulouse_AI4GEO2020-stereo_urban/testing/"
SaveDir="/home/qt/tengw/scratch/Toulouse_AI4GEO2020/experiment_echo_urban"
LIST="${data_path}toulouse_ai4geo_urban.txt"
#LIST="${data_path}toulouse_ai4geo_debug.txt"
INFO="/work/OT/ai4geo/users/tengw/stereobenchmark_AI4GEO/Toulouse_AI4GEO2020-stereo_urban/training/toulouse_ai4geo_info_rbg.txt"

python evaluate_color.py --model_path ${modelpath} \
--model_name deux2x \
--datalist ${LIST} \
--savepath ${SaveDir} \
--subfolder ${Method} \
--info_datapath ${INFO}
