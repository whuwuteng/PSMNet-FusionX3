#!/bin/bash

# run a folder example
BASEDIR=$(pwd -L)
echo "$BASEDIR"


modelname="gcnet_lidar"

modelpath="./pretrained/pre_trained_DublinCity_GCNet_CCVNorm.ckpt"

left="./example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_left.png"
right="./example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_right.png"
left_guide="./example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_left_guide.png"
right_guide="./example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_right_guide.png"
disp="./example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_res.png"

python evaluate_color_example.py --model_name ${modelname}  \
                                 --model_path ${modelpath} \
                                 --leftimg ${left} \
                                 --rightimg ${right} \
                                 --leftGuide ${left_guide} \
                                 --rightGuide ${right_guide} \
                                 --result ${disp}
