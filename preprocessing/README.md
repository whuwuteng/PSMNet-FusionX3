# Dataset preprocessing

## Introduction

For the dataset preprocessing, there are two parts : random sampling and Triangulated Irregular Network (TIN).

On the other handle, to keep the dataset small, we use the (dL-dR=d) to convert left disparity to right disparity.

## Left disparity to right

To minimize the dataset size,  for the ground truth (GT) disparity, we only keep the left GT disparity. In fact, the precision is lost during this converting. The python based code can be found in [folder](Left_right_disparity). There is an example :

```

#! /bin/bash

./create_right_disparity_list.sh

```

## Sampling

The original LiDAR is very dense, the sparse guidance is sampling from the dense GT disparity, the python base code can be found in [folder](Random_sampling). There is an example :

```

#! /bin/bash

./random_disparity.sh

```

## TIN interpolation

The TIN based interpolation uses the code from [2D triangulation](https://www.cs.cmu.edu/~quake/triangle.html), and the code uses C++, 

### Dependency

[Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)

[OpenImageIO](https://sites.google.com/site/openimageio/home) 

The dependency is installed by :

``` shell
#! /bin/bash

# install Eigen3 and other image library for openimageio
sudo apt-get install libeigen3-dev \
                     zlib1g-dev \
                     libboost-all-dev \
                     libtiff-dev \
                     libpng-dev \
                     libjpeg-dev \
                     
# install OpenImageIO
git clone --recurse https://github.com/OpenImageIO/oiio.git && \
cd oiio && \
mkdir build && \
cd build && \
cmake -DOIIO_BUILD_TESTS=0 -DUSE_OPENGL=0 -DUSE_QT=0 -DUSE_PYTHON=0 -DUSE_FIELD3D=0 -DUSE_FFMPEG=0 -DUSE_OPENJPEG=1 -DUSE_OPENCV=0 -DUSE_OPENSSL=0 -DUSE_PTEX=0 -DUSE_NUKE=0 -DUSE_DICOM=0 ../ && \
make -j4 && \
make install
```


### CMakeList for Ubuntu

``` shell
#! /bin/bash

# build
mkdir build
cd build
cmake ..
make

# example
./CreateDispDenseTIN ../example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_left_01.png  ../example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_left_01_tin.png ../example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_left_01_conf.png
```

## Prepare training data

Considering that we have published [two dataset](https://zenodo.org/record/8200023) :  DublinCity and Toulouse2020. We use the DublinCity as an example to show how to prepare the training data, the file structure of the folder is :

````
└── DublinCity-stereo_echo_new
    │
    ├── testing                                  # testing data
    │   ├── 3489_DUBLIN_...                      # stereo pair
    │   │   ├── colored_0                        # left images
    │   │   │   ├── 3489_DUBLIN_..._0000.png     # crop image 1
    │   │   │   ├── 3489_DUBLIN_..._0001.png     # crop image 2
    │   │   │   └── ...
    │   │   ├── colored_1                        # right images
    │   │   │   ├── 3489_DUBLIN_..._0000.png     # crop image 1
    │   │   │   ├── 3489_DUBLIN_..._0001.png     # crop image 2
    │   │   │   └── ...
    │   │   └── disp_occ                         # disparity images for the left image
    │   │       ├── 3489_DUBLIN_..._0000.png     # crop image 1
    │   │       ├── 3489_DUBLIN_..._0001.png     # crop image 2
    │   │       └── ...
    │   └── test_filelist_full.txt               # all the test folders
    ├── training                                 # training data
    │   ├── 3489_DUBLIN_...                      # stereo pair
    │   │   ├── colored_0                        # left images
    │   │   │   ├── 3489_DUBLIN_..._0000.png     # crop image 1
    │   │   │   ├── 3489_DUBLIN_..._0001.png     # crop image 2
    │   │   │   └── ...
    │   │   ├── colored_1                        # right images
    │   │   │   ├── 3489_DUBLIN_..._0000.png     # crop image 1
    │   │   │   ├── 3489_DUBLIN_..._0001.png     # crop image 2
    │   │   │   └── ...
    │   │   └── disp_occ                         # disparity images for the left image
    │   │       ├── 3489_DUBLIN_..._0000.png     # crop image 1 
    │   │       ├── 3489_DUBLIN_..._0001.png     # crop image 2
    │   │       └── ...
    │   ├── dublin_trainlist_full.txt            # image used for training (1200 images)
    │   └── dublin_vallist_full.txt              # image used for evaluation (200 images)
    └── Dublin_BHratio_ax.png                    # Base height ratio information
````

### Dataset for GuidedStereo

For this method, only the guidance of the left image is needed. In the following, I will give the command lines to prepare the data in **2.5%**. For the left image, the command lines are :

```
#! /bin/bash

# DATA_ROOT is the dataset folder

# generate the guidance for training data
python random_disparity_list.py --txtlist ${DATA_ROOT}/dublin_trainlist_full.txt --folder guide0_025 --scale 0.025

# generate the guidance for evaluation data
python random_disparity_list.py --txtlist ${DATA_ROOT}/dublin_vallist_full.txt --folder guide0_025 --scale 0.025
```

### Dataset for  GCNet-CCVNorm

For this method, the disparity of the right image is also need, after random sampling, to produce the right image, the  command lines are : 

```
#! /bin/bash

# DATA_ROOT is the dataset folder

# for the origin dense disparity(training)
python create_right_disparity_list.py --txtlist ${DATA_ROOT}/dublin_trainlist_full.txt --srcfolder disp_occ --tarfolder disp_occR --disp_scale 256

# for the guidance image, from left to right(training)
python create_right_disparity_list.py --txtlist ${DATA_ROOT}/dublin_trainlist_full.txt --srcfolder guide0_025 --tarfolder guideR0_025 --disp_scale 256

# for the origin dense disparity(evaluation)
python create_right_disparity_list.py --txtlist ${DATA_ROOT}/dublin_vallist_full.txt --srcfolder disp_occ --tarfolder disp_occR --disp_scale 256

# for the guidance image, from left to right(evaluation)
python create_right_disparity_list.py --txtlist ${DATA_ROOT}/dublin_vallist_full.txt --srcfolder guide0_025 --tarfolder guideR0_025 --disp_scale 256
```

### Dataset for PSMNet-FusionX3

For this method, the guidance need to be interpolation, after produce the guidance of right image, the  command lines are : 

```
#! /bin/bash

# DATA_ROOT is the dataset folder
# EXE_ROOT is the exe folder
cd ${DATA_ROOT}
ls -d */ | cut -f1 -d'/' > folder.txt

LIST=${DATA_ROOT}/folder.txt

Input="guide0_025"
Output="TIN0_025"

while read line; do

    echo ${line}
	PAIR=${line}

    outDir=${DATA_ROOT}/${PAIR}/${Output}/
    if [ ! -d ${outDir} ]; then
        mkdir ${outDir}
            echo "mkdir ${outDir}"
    fi

    cd ${DATA_ROOT}/${PAIR}/${Input}
    for file in `ls *.png`
    do
        filename="${file%.*}"
        src=${DATA_ROOT}/${PAIR}/${Input}/${file}
        tar=${outDir}/${file}
        cmp=${outDir}/${filename}_uncertainty.png

        if [ -f ${tar} ]; then
            echo ${tar} exists
        else
            cd ${EXE_ROOT}
            ./CreateDispDenseTIN ${src} ${tar} ${cmp}
        fi
    done
done < ${LIST}
```

## Feed Back

If you think you have any problem, contact Teng Wu <whuwuteng@gmail.com>
