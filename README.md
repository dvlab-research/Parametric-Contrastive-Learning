# Imbalanced Learning for Recognition
This repository contains the code of our papers on the topic of imbalanced learning for recognition.
* The code for our preprint paper "Region Rebalance for Long-Tailed Semantic Segmentation" ([paper](https://arxiv.org/pdf/2204.01969.pdf)) will be released soon;
* The code for our TPAMI 2022 paper "Residual Learning for Long-tailed recogntion" ([paper](https://arxiv.org/pdf/2101.10633.pdf) and [code](https://github.com/jiequancui/ResLT));
* The code for our ICCV 2021 paper "Parametric Contrastive Learning" ([paper](https://arxiv.org/pdf/2107.12028.pdf) and [code](https://github.com/dvlab-research/Parametric-Contrastive-Learning));




# Parametric-Contrastive-Learning
This repository contains the implementation code for ICCV2021 paper:  
**Parametric Contrastive Learning** (https://arxiv.org/abs/2107.12028)  

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/parametric-contrastive-learning/long-tail-learning-on-imagenet-lt)](https://paperswithcode.com/sota/long-tail-learning-on-imagenet-lt?p=parametric-contrastive-learning)    

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/parametric-contrastive-learning/long-tail-learning-on-inaturalist-2018)](https://paperswithcode.com/sota/long-tail-learning-on-inaturalist-2018?p=parametric-contrastive-learning)           

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/parametric-contrastive-learning/long-tail-learning-on-places-lt)](https://paperswithcode.com/sota/long-tail-learning-on-places-lt?p=parametric-contrastive-learning)    

If you find this code or idea useful, please consider citing our work:
```
@misc{cui2021parametric,
      title={Parametric Contrastive Learning}, 
      author={Jiequan Cui and Zhisheng Zhong and Shu Liu and Bei Yu and Jiaya Jia},
      year={2021},
      eprint={2107.12028},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Overview

In this paper, we propose Parametric Contrastive Learning (PaCo) to tackle long-tailed recognition. Based on theoretical analysis, we observe supervised contrastive loss tends to bias on high-frequency classes and thus increases the difficulty of imbalance learning. We introduce a set of parametric class-wise learnable centers to rebalance from an optimization perspective. Further, we analyze our PaCo loss under a balanced setting. Our analysis demonstrates that PaCo can adaptively enhance the intensity of pushing samples of the same class close as more samples are pulled together with their corresponding centers and benefit hard example learning. **Experiments on long-tailed CIFAR, ImageNet, Places, and iNaturalist 2018 manifest the new state-of-the-art for long-tailed recognition. On full ImageNet, models trained with PaCo loss surpass supervised contrastive learning across various ResNet backbones**.


# Results and Pretrained models
## Full ImageNet (Balanced setting)

 | Method | Model | Top-1 Acc(%) | link | log |  
 | :---: | :---: | :---: | :---: | :---: |
 | PaCo  | ResNet-50  | 79.3 | [download](https://drive.google.com/file/d/1fBbTWJlM3knjN0SIjrLhQw_TeUuoHsOe/view?usp=sharing) | [download](https://drive.google.com/file/d/1G_oTmBxAxwZdIa4YJSCJ-lYslwvsBMiF/view?usp=sharing) |
 | PaCo  | ResNet-101 | 80.9 | [download](https://drive.google.com/file/d/18lAQu33eN4pzhOi6lSmvfK6ORvD0gMGg/view?usp=sharing) | [download](https://drive.google.com/file/d/1axS5ryB-MjoKBRN4N9lVHyLQ2pOfKgMw/view?usp=sharing) |
 | PaCo  | ResNet-200 | 81.8 | [download](https://drive.google.com/file/d/14ZOI8tdUMGZFp08QfYog-aRr541psmur/view?usp=sharing) | [download](https://drive.google.com/file/d/18RNyK9HHSeQOQj69YUX_5Seq_PdGWavv/view?usp=sharing) | 
 
## ImageNet-LT (Imbalance setting)
 | Method | Model | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: |
 | PaCo  | ResNet-50   | 57.0 | [download](https://drive.google.com/file/d/1a73Ez_k47S2hmD_0L-sLH0OEhxK8SpQt/view?usp=sharing) | [download](https://drive.google.com/file/d/1NmynVzdkSye0FNEEyHSFF1oMc6q2wypJ/view?usp=sharing) |
 | PaCo  | ResNeXt-50  | 58.2 | [download](https://drive.google.com/file/d/1J7pvp-CWx7e2hPFNa1a05Oy9igHSe1eM/view?usp=sharing) | [download](https://drive.google.com/file/d/1ssvsA-xG2oj5wUwmC-Gu_pVZdg9POp7R/view?usp=sharing) |
 | PaCo  | ResNeXt-101 | 60.0 | [download](https://drive.google.com/file/d/1k14zhOwF8NBTb17mUN_UAGBkIIZsVBCV/view?usp=sharing) | [download](https://drive.google.com/file/d/1ZVwUKFb9AozaNKb8aSUXLCy27LgE7Kt2/view?usp=sharing) |
 
 ## iNaturalist 2018 (Imbalanced setting)
 | Method | Model | Top-1 Acc(%) | link |  log |
 | :---: | :---: | :---: | :---: | :---: |
 | PaCo  | ResNet-50   | 73.2 | TBD | [download](https://drive.google.com/file/d/1oYMqMcE9uC1pXwEOapB7zoha6Pjj7WO4/view?usp=sharing) |
 | PaCo  | ResNet-152  | 75.2 | TBD | [download](https://drive.google.com/file/d/1i5g10hlgNiPWOZ1zHn0wAhNag5F1ak9F/view?usp=sharing) |
 
 ## Places-LT (Imbalanced setting)
  | Method | Model | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: |
 | PaCo  | ResNet-152   | 41.2 | TBD | [download](https://drive.google.com/file/d/1kwu8AB5slPZLRm3OI-k3Jd6p22yMOThW/view?usp=sharing) |
 
# Get Started
For full ImageNet, ImageNet-LT, iNaturalist 2018, Places-LT training and evaluation. Note that PyTorch>=1.6. All experiments are conducted on 4 GPUs. **If you have more GPU resources, please make sure that the learning rate should be linearly scaled and 32 images per gpu is recommented**.
```
cd Full-ImageNet
bash sh/train_resnet50.sh
bash sh/eval_resnet50.sh

cd LT
bash sh/ImageNetLT_train_R50.sh
bash sh/ImageNetLT_eval_R50.sh
bash sh/PlacesLT_train_R152.sh
bash sh/PlacesLT_eval_R152.sh

cd LT
bash sh/CIFAR100_train_imb0.1.sh
```

# Contact
If you have any questions, feel free to contact us through email (jiequancui@link.cuhk.edu.hk) or Github issues. Enjoy!
 
 
 
 

