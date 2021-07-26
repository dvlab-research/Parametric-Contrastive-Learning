# Parametric-Contrastive-Learning
This repository contains the implementation code for paper:  
Parametric Contrastive Learning

# Overview
In this paper, we propose Parametric Contrastive Learning (PaCo) to tackle long-tailed recognition. Based on theoretical analysis, we observe supervised contrastive loss tends to bias on high-frequency classes and thus increases the difficulty of imbalance learning. We introduce a set of parametric class-wise learnable centers to rebalance from an optimization perspective. Further, we analyze our PaCo loss under a balanced setting. Our analysis demonstrates that PaCo can adaptively enhance the intensity of pushing samples of the same class close as more samples are pulled together with their corresponding centers and benefit hard example learning. Experiments on long-tailed CIFAR, ImageNet, Places, and iNaturalist 2018 manifest the new state-of-the-art for long-tailed recognition. On full ImageNet, models trained with PaCo loss surpass supervised contrastive learning across various ResNet backbones.

![image](https://github.com/jiequancui/Parametric-Contrastive-Learning/blob/main/assets/paco.PNG)



# Results and Pretrained models
## Full ImageNet

 | Method | Model | Top-1 Acc(%) | link |  
 | :---: | :---: | :---: | :---: |
 | PaCo  | ResNet-50  | 79.3 | [download]() |
 | PaCo  | ResNet-101 | 80.9 | [download]() |
 
## ImageNet-LT
 | Method | Model | Top-1 Acc(%) | link |  
 | :---: | :---: | :---: | :---: |
 | PaCo  | ResNet-50   | 57.0 | [download]() |
 | PaCo  | ResNeXt-50  | 58.2 | [download]() |
 | PaCo  | ResNeXt-101 | 60.0 | [download]() |
 
 ## iNaturalist 2018
 | Method | Model | Top-1 Acc(%) | link |  
 | :---: | :---: | :---: | :---: |
 | PaCo  | ResNet-50   | 73.2 | [download]() |
 | PaCo  | ResNet-152  | 75.2 | [download]() |
 
 ## Places-LT
  | Method | Model | Top-1 Acc(%) | link |  
 | :---: | :---: | :---: | :---: |
 | PaCo  | ResNet-152   | 41.2 | [download]() |
 
# Get Started
For full ImageNet, ImageNet-LT, iNaturalist 2018, Places-LT training and evaluation:  
```
cd Full-ImageNet
bash sh/train_resnet50.sh
bash sh/eval_resnet50.sh

cd LT
bash sh/ImageNetLT_train_R50.sh
bash sh/ImageNetLT_eval_R50.sh
bash sh/PlacesLT_train_R152.sh
bash sh/PlacesLT_eval_R152.sh
```

# Contact
If you have any questions, feel free to contact us through email (jiequancui@link.cuhk.edu.hk) or Github issues. Enjoy!
 
 
 
 

