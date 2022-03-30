# Learning Distinctive Margin toward Active Domain Adaptation
- A Pytorch implementation of our CVPR 2022 paper "Learning Distinctive Margin toward Active Domain Adaptation"
- [arXiv](https://arxiv.org/abs/2203.05738)

## Installation
- **Python 3.7**
- **Pytorch 1.8.0**
- **torchvision 0.9**
- **Numpy 1.20**


## Run the code

### Preliminaries

- Prepare dataset: [OfficeHome](http://hemanthdv.org/OfficeHome-Dataset/), [Office31](https://faculty.cc.gatech.edu/~judy/domainadapt/) and [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)
	+ We have provided text index files.


### Training

- Setting

Modify the configuration in `SDM_code/config/ini.config`
```
Arg:
[data]
name : dataset
path = dataset location
source = the initial of certain scenario 
target = the initial of certain scenario
class = number of categories
[sample]
strategy = certain sample strategy
[param]
epoch : we set it to 40 in our experiments
lr : learning rate
batch : batch size
sdm_lambda : default value is 0.01
sdm_margin : default value is 1.0
```

- Usage

After modify setting, just run the code:
```
python3 run.py
```

- Log

We also provide our experiment logs saved in `SDM_code/log/{dataset}_{source}{target}.log`. For example, `officehome_AC.log`


## Acknowledgement

This codebase is built upon [TQS](https://github.com/thuml/Transferable-Query-Selection).

## Citation
If you find our work helps your research, please kindly consider citing our paper in your publications.
```
@article{xie2022sdm
	title={Learning Distinctive Margin toward Active Domain Adaptation},
    author={Xie, Ming and Li, Yuxi and Wang, Yabiao and Luo, Zekun and Gan, Zhenye and Sun, Zhongyi and Chi, Mingmin and Wang, Chengjie and Wang, Pei},
    booktitle={IEEE/CVF International Conference on Computer Vision and Pattern Recognition},
    year={2022}
}
```
