# PSMNet-FusionX3 : LiDAR-Guided Deep Learning Stereo Dense Matching On Aerial Images

## Introduction

The site is for the paper appear in CVPR2023 [Photogrammetric Computer Vision Workshop](https://photogrammetric-cv-workshop.github.io/).

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

The data is generated using our [previous work](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B2-2021/405/2021/).
The training and testing splitting is :

|   <img src="/figures/DublinCity_show.jpg" width="700" alt="*DublinCity coverage*" />           |     <img src="/figures/Toulouse2020_show.jpg" width="700" alt="*Toulouse2020 coverage*" />           |
| :--------: | :----------: |
| DublinCity | Toulouse2020 |

## Method




## Feed Back

If you think you have any problem, contact Teng Wu <whuwuteng@gmail.com>



