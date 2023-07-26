# Guided Stereo Matching (revised)

This code is revised from the [official code](https://github.com/mattpoggi/guided-stereo), only the input and output is changed.



## Example

We will give an  example in the code folder, the file is shown in [example folder](./example), the pre-trained model is in [pre-trained folder](pretrained).

We can test the example by running :

```

#! /bin/bash

./evaluate_color_example.sh

```

The result is shown in : 

| <img src="example/3489_DUBLIN_AREA_2KM2_rgb_125019_id412c1_20150326121409_3489_DUBLIN_AREA_2KM2_rgb_128080_id728c1_20150326151319_0004_res.png" width="400"> |
| :----------------------------------------------------------: |
|     *The result of Guided stereo maching on DublinCity*      |

## Training

The training step depends on the structure of the training data, we will git an example in the file **train_toulouse_guide.sh**.

## Feed Back

If you think you have any problem, contact Teng Wu <whuwuteng@gmail.com>
