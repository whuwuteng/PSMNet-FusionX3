#!/bin/bash

# run a folder example
BASEDIR=$(pwd -L)
echo "$BASEDIR"

#modelname="/work/OT/ai4geo/users/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo/finetune_549_low.tar"

#modelname="/work/OT/ai4geo/users/tengw/DublinCity_Model_echo/guide_stereo/finetune_344.tar"

#modelname="/work/OT/ai4geo/users/tengw/Toulouse_TlseMetro_Model_echo/guide_stereo/finetune_649.tar"

#modelname="/work/scratch/tengw/DublinCity_Model_echo/guide_stereo0_005/finetune_559.tar"

#modelname="/work/scratch/tengw/DublinCity_Model_echo/guide_stereo0_01/finetune_789.tar"

#modelname="/work/scratch/tengw/DublinCity_Model_echo/guide_stereo0_025/finetune_459.tar"

#modelname="/work/scratch/tengw/DublinCity_Model_echo/guide_stereo0_05/finetune_464.tar"

#modelname="/work/OT/ai4usr/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo0_05/finetune_544.tar"

#modelname="/work/OT/ai4usr/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo0_025/finetune_564.tar"

#modelname="/work/OT/ai4usr/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo0_01/finetune_544.tar"

#modelname="/work/OT/ai4usr/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo/finetune_549.tar"

modelname="/work/OT/ai4usr/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo0_005/finetune_549.tar"

#modelname="/work/scratch/tengw/DublinCity_Model_echo/guide_stereo_pattern/finetune_639.tar"

Method="Toulouse2020_GuideStereo0_005_recal"
#Method="GuideStereo"

data_path="/work/scratch/tengw/stereodense_training_select/testing/DublinCity-stereo_echo_new/testing/"
SaveDir="/work/scratch/tengw/DublinCity/experiment_echo_lidar/"
LIST="${data_path}/dublin_test.txt"

python evaluate_color.py --loadmodel ${modelname}  \
--datalist ${LIST} \
--savepath ${SaveDir} \
--subfolder ${Method} \
--guidefolder guide0_005 \
--guided
