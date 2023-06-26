#! /bin/bash
#--loadmodel /work/OT/ai4geo/users/tengw/PreTrain_Model_Save/PSMNet/pretrained_model_KITTI2015.tar \


BASEDIR=$(pwd -L)
echo "${BASEDIR}"

# data copy
echo ${TMPDIR}
cd ${TMPDIR}
cp -r /work/OT/ai4geo/users/tengw/stereodense_training_select/Toulouse_AI4GEO2020-stereo_urban_guide/training .

DATA=${TMPDIR}

cd ${BASEDIR}

#DATA=/work/OT/ai4geo/users/tengw/stereodense_training_select/Toulouse_AI4GEO2020-stereo_urban_guide

python finetune_vaihingen_ccvnorm.py --maxdisp 192 \
--train_datapath ${DATA}/training/toulouse_ai4geo_trainlist_guide.txt \
--val_datapath ${DATA}/training/toulouse_ai4geo_vallist_guide.txt \
--info_datapath ${DATA}/training/toulouse_ai4geo_info_rbg.txt \
--model full \
--epochs 800 \
--savemodel /work/OT/ai4geo/users/tengw/Toulouse_AI4GEO2020_Model_echo/PSMnet_LiDAR_full/

#--resume \