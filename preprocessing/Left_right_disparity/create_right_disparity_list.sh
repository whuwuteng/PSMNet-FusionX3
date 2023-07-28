#! /bin/bash

#DATA_ROOT="/work/OT/ai4geo/users/tengw/stereobenchmark_AI4GEO/Toulouse_AI4GEO2020-stereo_urban/training"
#DATA_ROOT="/work/OT/ai4geo/users/tengw/stereobenchmark_AI4GEO/Toulouse_AI4GEO2020-stereo_urban/testing"

#DATA_ROOT="/work/OT/ai4geo/users/tengw/stereobenchmark_AI4GEO/DublinCity-stereo_echo_new/training"
#DATA_ROOT="/work/OT/ai4geo/users/tengw/stereobenchmark_AI4GEO/DublinCity-stereo_echo_new/testing"

#DATA_ROOT="/work/OT/ai4geo/users/tengw/stereobenchmark_AI4GEO/Toulouse_TlseMetro-stereo_echo_new/training"
#DATA_ROOT="/work/OT/ai4geo/users/tengw/stereodensematchingbenchmark/Toulouse_TlseMetro-stereo_echo/testing"

#DATA_ROOT="/work/OT/ai4geo/users/tengw/stereodense_training_select/IARPA-stereo_echo_new/training"

#DATA_ROOT="/work/scratch/tengw/stereodense_training_select/training/DublinCity-stereo_echo_new/training"


DATA_ROOT="/work/scratch/tengw/stereodense_training_select/testing/DublinCity-stereo_echo_new/testing"

python create_right_disparity_list.py --txtlist ${DATA_ROOT}/dublin_test.txt \
                                      --srcfolder guide0_01 \
                                      --tarfolder guideR0_01 \
                                      --disp_scale 256


