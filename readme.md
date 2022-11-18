# ResNet-Pytorch

## Description

Pytoch Implementation of ResNet and train script with ImageNet and Caltech-256.

## Install

```
git clone https://github.com/mktj2685/ResNet-PyTorch.git
cd ResNet-Pytorch
pip install -r requirements.txt
pip install -e .
```

## Usage

1. Please download and unzip [ImageNet](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description) or [Caltech-256](https://www.kaggle.com/datasets/jessicali9530/caltech256) at `ResNet-Pytorch/data/imagenet` or `ResNet-Pytorch/data/caltech256`.

```
ResNet-Pytorch/
    └ data/
        ├ ImageNet/
        │   ├ ILSVRC/
        │   └ LOC_synset_mapping.txt
        │
        └ Caltech256/
            └ 256_ObjectCategories/
```

2. execute train script.
```
python tools/train.py resnet-18 imagenet --epoch 100 --batch_size 64
```

## Reference

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and
Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015.