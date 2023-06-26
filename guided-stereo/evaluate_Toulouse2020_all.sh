#!/bin/bash

# run a folder example
BASEDIR=$(pwd -L)
echo "$BASEDIR"

modelname="/work/OT/ai4usr/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo/finetune_549.tar"

#modelname="/work/OT/ai4geo/users/tengw/DublinCity_Model_echo/guide_stereo/finetune_344.tar"

#modelname="/work/OT/ai4usr/tengw/Toulouse_TlseMetro_Model_echo/guide_stereo/finetune_649.tar"

#modelname="/work/OT/ai4usr/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo0_025/finetune_564.tar"

#modelname="/work/OT/ai4usr/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo0_01/finetune_544.tar"

#modelname="/work/OT/ai4usr/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo0_05/finetune_544.tar"

#modelname="/work/OT/ai4usr/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo0_005/finetune_549.tar"

#modelname="/work/scratch/tengw/DublinCity_Model_echo/guide_stereo0_05/finetune_464.tar"

#modelname="/work/scratch/tengw/DublinCity_Model_echo/guide_stereo0_025/finetune_459.tar"

#modelname="/work/scratch/tengw/DublinCity_Model_echo/guide_stereo0_01/finetune_789.tar"

#modelname="/work/scratch/tengw/DublinCity_Model_echo/guide_stereo0_005/finetune_559.tar"

#modelname="/work/OT/ai4usr/tengw/DublinCity_Model_echo/guide_stereo/finetune_344.tar"

#modelname="/work/scratch/tengw/Toulouse_AI4GEO2020_Model_echo/guide_stereo_pattern/finetune_604.tar"

Method="GuideStereo_Toulouse2020"

#Method="GuideStereo_Tlse"

data_path="/work/scratch/tengw/stereodense_training_select/testing/Toulouse_AI4GEO2020-stereo_urban/testing/"
SaveDir="/work/scratch/tengw/Toulouse_AI4GEO2020/experiment_echo_urban"
#LIST="${data_path}toulouse_ai4geo_debug.txt"
LIST="${data_path}toulouse_ai4geo_urban.txt"

python evaluate_color.py --loadmodel ${modelname}  \
--datalist ${LIST} \
--savepath ${SaveDir} \
--subfolder ${Method} \
--guidefolder guide \
--guided

