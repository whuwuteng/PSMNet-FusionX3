# GCNet_CCVNorm (revised)

This code is revised from the [official code](https://github.com/zswang666/Stereo-LiDAR-CCVNorm), only the input and output is changed.

## Introduction

Because the GCNet doesn't provide the official code, the base method is GCNet, so in the paper, we use this version of GCNet.

## Example

### GCNet

#### Testing

We will give an  example in the code folder, the file is shown in [example folder](./example), the pre-trained model is in [pre-trained folder](pretrained).

We can test the example by running :

```

#! /bin/bash

./evaluate_color_example_GCNet.sh

```

The result can be shown : 
| <img src="example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_res.png" width="400"> |
| :----------------------------------------------------------: |
|             *The result of GCNet on DublinCity*              |

## GCNet_CCVNorm 

#### Testing

We will give an  example in the code folder, the file is shown in [example folder](./example), the pre-trained model is in [pre-trained folder](pretrained).

We can test the example by running :

```

#! /bin/bash

./evaluate_color_example_GCNet_CCVNorm.sh

```

In the example, the guidance is 0.5%, the result can be show :

| <img src="example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_res_guide.png" width="400"> |
| :----------------------------------------------------------: |
|         *The result of GCNet-CCVNorm on DublinCity*          |



#### Training

The training step depends on the structure of the training data, we will git an example in the file **train_toulouse_guide.sh**.

