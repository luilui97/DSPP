# Feature Decoupled Knowledge Distillation via Spatial Pyramid Pooling

## Introduction

Knowledge distillation (KD) is an effective and widely used technique of model compression which enables the deployment of deep networks in low-memory or fast-execution scenarios. Feature-based knowledge distillation is an important component of KD which leverages intermediate layers to supervise the training procedure of a student network. Nevertheless, the potential mismatch of intermediate layers may be counterproductive in the training procedure. In this paper, we propose a novel distillation framework, termed Decoupled Spatial Pyramid Pooling Knowledge Distillation, to distinguish the importance of regions in feature maps. Specifically, we reveal that (1) spatial pyramid pooling is an outstanding method to define the knowledge and  (2) the lower activation regions in feature maps play a more important role in KD. Our experiments on CIFAR-100 and Tiny-ImageNet achieve state-of-the-art results.

The work is accepted by the 16th Asian Conference on Computer Vision， ACCV2022.

This repo contains the code of DSPP and other classical KD methods including FitNet, AT, SP, CRD, et al.

## Overview

We propose Decoupled Spatial Pyramid Pooling Knowledge Distillation (DSPP) to exploit intermediate knowledge by exploring a novel method to define knowledge and decoupling feature map to optimize KD training procedure. A spatial pyramid pooling architecture is applied in our approach for automatically perceiving knowledge, which effectively captures informative knowledge at various scales of feature map. Then a decoupling module is designed to analyze region-level semantic loss between student and teacher network based on the observation that the lower activation regions in feature map plays a more important role in KD, i.e., lower activation regions contain more informative knowledge cues. To align the spatial dimension of teacher and student layer pair, feature map of the student layer is projected to the same dimension of the teacher layer. By taking advantage of spatial pyramid pooling and decoupled region-level loss assignment, the student network can be effectively optimized with more sophisticated supervision. 

![DSPP/overview.png at main · luilui97/DSPP (github.com)](https://github.com/luilui97/DSPP/blob/main/image/overview.png)

## Results

![DSPP/results.png at main · luilui97/DSPP (github.com)](https://github.com/luilui97/DSPP/blob/main/image/results.png)

## Running

```bash
# Cifar 100
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill SPP --model_s resnet8x4 -r 1 -a 1 -b 1 --topk 100 --hint_layer 3 --spp_layer 3 --dataset cifar100 --min_w 10 --max_w 1 --trial 0 
```

- `distill`: specify the KD method
- `path_t`: specify the pretrained model of teacher network
- `model_s`: specify the student network
- `dataset`: specify the training dataset
- `r`: specify the weight for classification
- `a`: specify the weight for vanilla KD
- `b`: specify the weight for the KD method specified by `distill`
- `topk`: specify the larger activation part
- `hint_layer`: specify the last hint layer of the teacher
- `spp_layer`: specify number of spp layers
- `min_w` and `max_w`: specify the weights for smaller and larger activation regions respectively

## Note

The implementation of compared methods are based on the author-provided code and an open-source benchmark https://github.com/HobbitLong/RepDistiller. 

