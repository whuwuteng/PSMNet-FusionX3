# PSMNet-FusionX3 : LiDAR-Guided Deep Learning Stereo Dense Matching On Aerial Images

## Introduction

The site is for the paper appear in CVPR2023 [Photogrammetric Computer Vision Workshop](https://photogrammetric-cv-workshop.github.io/program.html).

The 8 minute presentation can be found [here](https://docs.google.com/presentation/d/1Phdm8ITpiKVgrHSTDY7iRewfC_6cToJ3z4dt_W6dZ3Y/edit?usp=sharing).

I also make a poster, but I did not participate in person because of visa issue, the poster can be found  [here](https://docs.google.com/presentation/d/1i2DSw-lg7Hk6dbXfVvOf2UsuL5RfdTL_/edit?usp=sharing&ouid=108677916770799835536&rtpof=true&sd=true).

## Dataset

In the paper, we use two dataset with high dense LiDAR.

|   dataset    | Image GSD(cm) | LiDAR density( $pt/m^2$ ) |
| :----------: | :-----------: | :-----------------------: |
|  DublinCity  |      3.4      |          250-348          |
| Toulouse2020 |       5       |            50             |

### DublinCity Dataset

[DublinCity](https://v-sense.scss.tcd.ie/dublincity/) is an open dataset, the original aerial and LiDAR point cloud can be [downloaded](https://geo.nyu.edu/catalog/nyu-2451-38684), the origin dataset is very large.

| <img src="/figures/DublinCity.png" width="700" alt="*Origin DublinCity coverage*" /> |
| :----------------------------------------------------------: |
|                *Origin DublinCity coverage*                |

Because the origin dataset use Terrasolid for the orientation, the origin orientation is not accurate, so the registration step is mondatory, the experiment area is shown in 

| <img src="/figures/DublinCity_cover.png" width="700" alt="*DublinCity coverage*" /> |
| :----------------------------------------------------------: |
|                *DublinCity experiment  coverage*                |

### Toulouse2020 Dataset

Toulouse2020 is a  dataset collected by [IGN (French Mapping Agency)](https://www.ign.fr/) for [AI4GEO project](https://www.ai4geo.eu/), the origin dataset is very large.

| <img src="/figures/Toulouse.jpg" width="700" alt="*Origin Toulouse2020 coverage*" /> |
| :----------------------------------------------------------: |
|                *Origin Toulouse2020 coverage*                |

Because the whole area is too large, in order to  registration the image and LiDAR, we select the center city of Toulouse, the experiment area is shown in 
| <img src="/figures/Toulouse2020_cover.png" width="700" alt="*Toulouse2020 coverage*" /> |
| :----------------------------------------------------------: |
|                *Toulouse2020 experiment  coverage*                |

### Data set generation

The data is generated using our [previous work](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B2-2021/405/2021/), the detail introduction can also be found on [Github](https://github.com/whuwuteng/benchmark_ISPRS2021).
The training and testing splitting is :

|   <img src="/figures/DublinCity_show.jpg" width="700" alt="*DublinCity coverage*" />           |     <img src="/figures/Toulouse2020_show.jpg" width="700" alt="*Toulouse2020 coverage*" />           |
| :--------: | :----------: |
| DublinCity | Toulouse2020 |

We will also publish the dataset for public use, because the original dataset is too large, at present, we will only publish the used training and testing dataset in the paper.

All the dataset are host by [Zenodo](https://zenodo.org/), the download site is here.

## Method

In the paper, we propose a method based on PSMNet[^1],  based on  the stereo work, like the work[^2] in computer vision, we use the TIN expansion for remote sensing data. The newwork is shown : 

| <img src="/figures/PSMNet_LiDAR.jpg" width="700" alt="*PSMNet-FusionX3*" /> |
| :----------------------------------------------------------: |
|                *PSMNet-FusionX3*                |

In the experiment, we compare our method with GCNet[^3], PSMNet, GuidedStereo[^4], and GCNet-CCVNorm[^5]. There is no official code for GCNet, so the GCNet result is from the code of GCNet-CCVNorm.


[^1]: Chang, Jia-Ren, and Yong-Sheng Chen. "Pyramid stereo matching network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[^2]: Huang, Yu-Kai, et al. "S3: Learnable sparse signal superdensity for guided depth estimation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

[^3]: Kendall, Alex, et al. "End-to-end learning of geometry and context for deep stereo regression." Proceedings of the IEEE international conference on computer vision. 2017.

[^4]: Poggi, Matteo, et al. "Guided stereo matching." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

[^5]: Wang, Tsun-Hsuan, et al. "3d lidar and stereo fusion using stereo matching network with conditional cost volume normalization." 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019.

## Code

PSMNet-FusionX3 and the methods compared in the paper are all provided here.

The pre-trained  models will be also available.

For the other methods, because  our dataset is different from the computer vision dataset, we will also put the revised code in this repository. 

### Guided Stereo Matching

We revise the [official code](https://github.com/mattpoggi/guided-stereo)  to adopt to the remote sensing dataset, the detail can be found in [folder](./guided-stereo).

### GCNet-CCVNorm

We revise the [official code](https://github.com/zswang666/Stereo-LiDAR-CCVNorm)  to adopt to the remote sensing dataset, the detail can be found in [folder](./GCNet-CCVNorm).

### PSMNet-FusionX3 

We also release the code of our method, the detail can be found in [folder](./PSMNet-FusionX3).

### Dataset proprocessing

Because the input guidance is sampled from the origin dense disparity, and for the TIN based interpolation, this is processed before training, the detail can be found in [folder](./preprocessing).

## BibTeX Citation

<pre>@inproceedings{wu2023psmnet,
  title={PSMNet-FusionX3: LiDAR-Guided Deep Learning Stereo Dense Matching on Aerial Images},
  author={Wu, Teng and Vallet, Bruno and Pierrot-Deseilligny, Marc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6526--6535},
  year={2023}
}
</pre>

## Feed Back

If you think you have any problem, contact Teng Wu <whuwuteng@gmail.com>



