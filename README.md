# (NeurlPS 2022) Prototypical VoteNet for Few-Shot 3D Point Cloud Object Detection  

## Environments

First you have to make sure that you have installed all dependencies. 

```
conda create --name prototypical_votenet python=3.8 -y
conda activate prototypical_votenet
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install mmdet
pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install setuptools==59.5.0
pip install mmsegmentation
pip install -e . 
```

Our implementation has been tested on one NVIDIA 3090 GPU with cuda 11.2.

## Contact
If you have any questions, you can email me (zhaosz@eee.hku.hk).
