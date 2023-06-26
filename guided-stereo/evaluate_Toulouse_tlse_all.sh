#!/bin/bash

# run a folder example
BASEDIR=$(pwd -L)
echo "$BASEDIR"

modelname="/work/OT/ai4geo/users/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo/finetune_549_low.tar"

#modelname="/work/OT/ai4geo/users/tengw/DublinCity_Model_echo/guide_stereo/finetune_344.tar"

#modelname="/work/OT/ai4geo/users/tengw/Toulouse_TlseMetro_Model_echo/guide_stereo/finetune_649.tar"

#Method="GuideStereo_549"
Method="GuideStereo_Toulouse2020"

data_path="/work/OT/ai4geo/users/tengw/stereodensematchingbenchmark/Toulouse_TlseMetro-stereo_echo/testing/"
SaveDir="/home/qt/tengw/scratch/Toulouse_TlseMetro/experiment_echo/"
LIST="${data_path}toulouse_tlse_test.txt"
#LIST="${data_path}toulouse_tlse_test_debug.txt"

python evaluate_color.py --loadmodel ${modelname}  \
--datalist ${LIST} \
--savepath ${SaveDir} \
--subfolder ${Method} \
--guided
