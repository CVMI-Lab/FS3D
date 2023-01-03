# Prototypical VoteNet for Few-Shot 3D Point Cloud Object Detection

<p align="center">
    <a href="https://nips.cc/virtual/2022/poster/55053"><img src="https://img.shields.io/badge/-NeurIPS%202022-68488b"></a>
    <a href="https://arxiv.org/abs/2210.05593"><img src="https://img.shields.io/badge/arXiv-2210.05593-b31b1b"></a>
  <a href="https://github.com/CVMI-Lab/SlotCon/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>
<p align="center">
	Prototypical VoteNet for Few-Shot 3D Point Cloud Object Detection (NeurIPS 2022)<br>
  By
  Shizhen Zhao
  and 
  Xiaojuan Qi.
</p>

## Introduction

Most existing 3D point cloud object detection approaches heavily rely on large amounts of labeled training data. However, the labeling process is costly and time-consuming. This paper considers few-shot 3D point cloud object detection, where only a few annotated samples of novel classes are needed with abundant samples of base classes. To this end, we propose Prototypical VoteNet to recognize and localize novel instances, which incorporates two new modules: Prototypical Vote Module (PVM) and Prototypical Head Module (PHM). Specifically, as the 3D basic geometric structures can be shared among categories, PVM is designed to leverage class-agnostic geometric prototypes, which are learned from base classes, to refine local features of novel categories.Then PHM is proposed to utilize class prototypes to enhance the global feature of each object, facilitating subsequent object localization and classification, which is trained by the episodic training strategy. To evaluate the model in this new setting, we contribute two new benchmark datasets, FS-ScanNet and FS-SUNRGBD. We conduct extensive experiments to demonstrate the effectiveness of Prototypical VoteNet, and our proposed method shows significant and consistent improvements compared to baselines on two benchmark datasets.

<p align="center">
	<img src=pic/method.png width=80% />
<p align="center">

	

## Environments

Please make sure that you have installed all dependencies. Our implementation has been tested on one NVIDIA 3090 GPU with cuda 11.2.

**Step 1.** (Create virtual env using conda)
```
conda create --name prototypical_votenet python=3.8 -y
conda activate prototypical_votenet
```

**Step 2.** (Intall [Pytorch](https://pytorch.org/))
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
You may change the above command according to your cuda version. Please refer to [official website](https://pytorch.org/) of Pytorch. 


**Step 3.** (Install mmdet, mmcv and mmsegmentation)
```
pip install mmdet==2.19.0
pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmsegmentation==0.20.0
```
You may change the above command according to your Pytorch version. Please refer to [official website](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md) of MMDetection3D. 


**Step 4.** (Setup the code base in your env)
```
pip install setuptools==58.0.4
pip install -v -e .
```


## Dataset Preparation

Request the datasets (FS-ScanNet and FS-SUNRGBD) from zhaosz@eee.hku.hk (academic only). Due to licensing issues, please send me your request using your university email.

After downloading FS-ScanNet and FS-SUNRGBD, you should unzip and put it in your project folder. The datasets have been processed so that you can directly use them to train your own models. 


## Train and Test the Model

For example, train and test the model on FS-SUNRGBD 1-shot, 2-shot, 3-shot, 4-shot, and 5-shot. 
```
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
./configs/prototypical_votenet/train_together_sun/prototypical_votenet_16x8_sunrgbd-3d-10class_1_1.py --sample_num 16 --work_path work_path/sunrgbd_split1_shot1_1

CUDA_VISIBLE_DEVICES=1 python tools/train.py \
./configs/prototypical_votenet/train_together_sun/prototypical_votenet_16x8_sunrgbd-3d-10class_1_2.py --sample_num 16 --work_path work_path/sunrgbd_split1_shot2_1

CUDA_VISIBLE_DEVICES=2 python tools/train.py \
./configs/prototypical_votenet/train_together_sun/prototypical_votenet_16x8_sunrgbd-3d-10class_1_3.py --sample_num 16 --work_path work_path/sunrgbd_split1_shot3_1

CUDA_VISIBLE_DEVICES=3 python tools/train.py \
./configs/prototypical_votenet/train_together_sun/prototypical_votenet_16x8_sunrgbd-3d-10class_1_4.py --sample_num 16 --work_path work_path/sunrgbd_split1_shot4_1

CUDA_VISIBLE_DEVICES=4 python tools/train.py \
./configs/prototypical_votenet/train_together_sun/prototypical_votenet_16x8_sunrgbd-3d-10class_1_5.py --sample_num 16 --work_path work_path/sunrgbd_split1_shot5_1
```

You can find the commands in the folder named "tools/".

## ModelZoo
### FS-ScanNet 
#### Split-1

| 1-shot             | 3-shot             | 5-shot             |
|--------------------|--------------------|--------------------|
| [model](https://drive.google.com/file/d/1WTnSUCdJu_dkwb7jjLcoR4791kgyUChW/view?usp=sharing) | [model](https://drive.google.com/file/d/1oJAS1x91ICPCNEEFvOKRFnO1TwOjWNXA/view?usp=sharing) | [model](https://drive.google.com/file/d/1QtBBrdyAxKiUBP3ZhGB8CRCDkRWXUF4H/view?usp=sharing)|

#### Split-2

| 1-shot             | 3-shot             | 5-shot             |
|--------------------|--------------------|--------------------|
| [model](https://drive.google.com/file/d/1UVCJRm-ABbqTcWt5g4r8EqfuANUjFpYT/view?usp=sharing) | [model](https://drive.google.com/file/d/1hAXfdyYE4zD5vaTZf_gWGBfi1pGA3XJF/view?usp=sharing)  | [model](https://drive.google.com/file/d/1B_oN9ogbmGIN-he6dgKbOkPS86Eyuqpi/view?usp=sharing) |

### FS-SUNRGBD

| 1-shot             | 2-shot             | 3-shot             | 4-shot             | 5-shot             |
|--------------------|--------------------|--------------------|--------------------|--------------------|
| [model](https://drive.google.com/file/d/16omLmv1laapqv4mGrER3LEPzeFerG7FI/view?usp=sharing)| [model](https://drive.google.com/file/d/1ohbk8efkYrMjOlBCHx_U-SXOyStrU5Pf/view?usp=sharing) | [model](https://drive.google.com/file/d/1h9OhzPwgR9xoVnw8-59udW23-u06M-iO/view?usp=sharing)| [model](https://drive.google.com/file/d/1qpUVrTsUEzNhWOsshUu-GSCb6rtbPCDy/view?usp=sharing)  | [model](https://drive.google.com/file/d/1N1pEqmcfC2bIC1GFuOl94HnOdRKgYq3q/view?usp=sharing) |

## Citation
Please consider :grimacing: staring this repository and citing the following paper if you feel this repository useful. 

```
@inproceedings{zhao2022fs3d,
  title={Prototypical VoteNet for Few-Shot 3D Point Cloud Object Detection},
  author={Zhao, Shizhen and Qi, Xiaojuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```


## Acknowledgement

Our code is largely based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), and we thank the authors for their implementation. Please also consider citing their wonderful code base. 

```
@misc{mmdet3d2020,
    title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},
    author={MMDetection3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
    year={2020}
}
```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
	
## Contact
If you have any questions, you can email me (zhaosz@eee.hku.hk). 
