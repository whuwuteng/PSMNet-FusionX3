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

# install Eigen3
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

## Feed Back

If you think you have any problem, contact Teng Wu <whuwuteng@gmail.com>
