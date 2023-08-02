# PSMNet-FusionX3

## Introduction

In the folder, the code of PSMNet-FusionX3 is provide. In the [paper](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Wu_PSMNet-FusionX3_LiDAR-Guided_Deep_Learning_Stereo_Dense_Matching_on_Aerial_Images_CVPRW_2023_paper.html),  Pytorch and [PytorchLightning](https://www.pytorchlightning.ai/index.html) are used. In the following, the training and testing part are introduced.

## Training

Before training, the training data need to be prepare before training, the detail can be found in folder [preprocessing](../preprocessing).



## Testing



We will give an  example in the code folder, the file is shown in [example folder](./example), the pre-trained model is in [pre-trained folder](pretrained).

We can test the example by running :

```

#! /bin/bash

./evaluate_color_example.sh

```

The result is shown in : 



## PytorchLightning




## Acknowledgments

Thanks to Jia-Ren Chang for sharing the original implementation of PSMNet: https://github.com/JiaRenChang/PSMNet.

Thanks to [Matteo Poggi](https://vision.disi.unibo.it/~mpoggi/) for  sharing the original implementation of Guided Stereo Matching : https://github.com/mattpoggi/guided-stereo.

Thanks to Tsun-Hsuan Wang for sharing the origin implementation of Stereo-LiDAR-CCVNorm : https://github.com/zswang666/Stereo-LiDAR-CCVNorm.

## Feed Back

If you think you have any problem, contact Teng Wu <whuwuteng@gmail.com>

