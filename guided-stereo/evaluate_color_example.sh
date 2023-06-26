#!/bin/bash

BASEDIR=$(pwd -L)
echo "$BASEDIR"


modelname="./pretrained/pre_trained_DublinCity_guide0_05.tar"

left="./example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_left.png"
right="./example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_right.png"
guide="./example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_guide.png"
disp="./example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_res.png"

python evaluate_color_example.py --loadmodel ${modelname}  \
--left ${left} \
--right ${right} \
--guide ${guide} \
--disp ${disp}