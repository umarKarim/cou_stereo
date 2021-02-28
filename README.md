# Towards Continual, Online, Unsupervised Depth  
## Introduction 
This is the source code for the paper **Towards Continual, Online, Unsupervised Depth**. This code is for stereo-based depth estimation. The SfM-based depth estimation is also available. 

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

## Pre-Training 
Set paths in the *dir_options/pretrain_options.py* file. Then run 

```
python pretrain.py
```
The pre-trained models should be saved in the directory *trained_models/pretrained_models/*.

## Online Training 
Set paths in the *dir_options/online_train_options.py* file. Then run 

```
python script_online_train.py
```
The online-trained models (for a single epoch only) will be saved in the *trained_models* directory. Intermediate results will be saved in the *qual_dmaps* directory. 

## Testing 
Set paths in *dir_options/test_options.py* file. Then run

```
python script_test_directory.py
```

Results will be stored in the *results* directory. Then run 

``` 
python script_evaluate.py
```

Results will be displayed in the console.

## Results 
Quantitative results are mentioned in the manuscript. Check the following [video](https://www.youtube.com/watch?v=_WNYOTDaCCM&t=10s&ab_channel=Depth) for qualitative results. 




