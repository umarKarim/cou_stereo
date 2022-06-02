# Towards Continual, Online, Unsupervised Depth  
## Introduction 
This is the source code for the paper **Towards Continual, Online, Unsupervised Depth**. This code is for stereo-based depth estimation. 

Manuscript is available [here](https://arxiv.org/abs/2103.00369).

The SfM-based depth estimation is also available at [here](https://github.com/umarKarim/cou_sfm). 

## Requirements 
- PyTorch 
- Torchvision 
- NumPy 
- Matplotlib 
- OpenCV
- Torchvision
- Pandas 

## Data Preparation 
 Download the raw KITTI dataset, the Virtual KITTI [RGB](http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar), the KITTI test [dataset](https://1drv.ms/u/s!AiV6XqkxJHE2kz5Zy7jWZd2GyMR2?e=kBD4lb), and Virtual KITTI [depth](http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar). Extract data to appropriate locations. Saving SSD is encouraged but not required.


## Testing 
Run 

``` 
python script_evaluate.py
```

to display results in console. Run 

```
python script_test_directory.py
```

to do the evaluations again. 


## Online Training 
Set paths in the *options/online_train_options.py* file. Then run 

```
python script_online_train.py
```
The online-trained models (for a single epoch only) will be saved in the *trained_models* directory. Intermediate results will be saved in the *qual_dmaps* directory. 

## Pre-Training 
Set paths in the *options/pretrain_options.py* file. Then run 

```
python pretrain.py
```
The pre-trained models should be saved in the directory *trained_models/pretrained_models/*.



## Results 
Check this [video](https://www.youtube.com/watch?v=_WNYOTDaCCM&t=10s&ab_channel=Depth) for qualitative results.

The Absolute Relative metric is shown in the following table.

| Training Dataset | Approach | Current Dataset | Other Dataset |
| -------------- | ------------ | ------------ | -------------- |
KITTI | Fine-tuning | 0.1920 | 0.1980 |
KITTI | Proposed | 0.1825 | 0.1660 |
VKITTI | Fine-tuning | 0.1991 | 0.2090 |
VKITTI | Proposed | 0.1653 | 0.1770 |

See the following figure for comparison.

![figs directory](https://github.com/umarKarim/cou_stereo/blob/main/figs/kitti_vkitti_qual_crop.jpg)







