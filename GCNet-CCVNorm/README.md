# GCNet_CCVNorm (revised)

This code is revised from the [official code](https://github.com/zswang666/Stereo-LiDAR-CCVNorm), only the input and output is changed.

## Introduction

Because the GCNet doesn't provide the official code, the base method is GCNet, so in the paper, we use this version of GCNet.

## Example

### GCNet

#### Testing





## GCNet_CCVNorm 

#### Testing

We will give an  example in the code folder, the file is shown in [example folder](./example), the pre-trained model is in [pre-trained folder](pretrained).

We can test the example by running :

```

#! /bin/bash

./evaluate_color_example.sh

```



#### Training

The training step depends on the structure of the training data, we will git an example in the file **train_toulouse_guide.sh**.

