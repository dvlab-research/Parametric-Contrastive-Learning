# Imbalanced Learning for Recognition
This repository contains the code of our papers on the topic of imbalanced learning for recognition.
* Our paper ["Decoupled Kullback-Leibler (DKL) Divergence Loss"](https://arxiv.org/pdf/2305.13948v1.pdf) is accepted by **NeurIPS 2024**
* Our new paper ["Classes Are Not Equal: An Empirical Study on Image Recognition Fairness"](https://arxiv.org/abs/2402.18133) is accepted by **CVPR 2024**
* Our new arXiv paper ["Decoupled Kullback-Leibler (DKL) Divergence Loss"](https://arxiv.org/pdf/2305.13948v1.pdf) achieves new state-of-the-art on **knowledge distillation** and **adversarial robustness**. [Code](https://github.com/jiequancui/DKL) is released.
* Code for RR & CeCo is partially released.
* Our paper "Generalized Parametric Contrastive Learning" is accepted by **TPAMI 2023**.
* Our paper "Understanding Imbalanced Semantic Segmentation Through Neural Collapse" is accepted by **CVPR2023**. The code will be released soon.   
* The code for our preprint paper "Generalized Parametric Contrastive Learning" is released;
* The code for our preprint paper "Region Rebalance for Long-Tailed Semantic Segmentation" ([paper](https://arxiv.org/pdf/2204.01969.pdf)) will be released soon;
* The code for our **TPAMI 2022 paper "Residual Learning for Long-tailed recogntion"** ([paper](https://arxiv.org/pdf/2101.10633.pdf) and [code](https://github.com/jiequancui/ResLT));
* The code for our **ICCV 2021 paper "Parametric Contrastive Learning"** ([paper](https://arxiv.org/pdf/2107.12028.pdf) and [code](https://github.com/dvlab-research/Parametric-Contrastive-Learning));



# Generalized Parametric-Contrastive-Learning
This repository contains the implementation code for ICCV2021 paper **Parametric Contrastive Learning** (https://arxiv.org/abs/2107.12028) 
and TPAMI 2023 paper **Generalized Parametric Contrastive Learning** (https://arxiv.org/abs/2209.12400).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalized-parametric-contrastive-learning/long-tail-learning-on-inaturalist-2018)](https://paperswithcode.com/sota/long-tail-learning-on-inaturalist-2018?p=generalized-parametric-contrastive-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalized-parametric-contrastive-learning/long-tail-learning-on-places-lt)](https://paperswithcode.com/sota/long-tail-learning-on-places-lt?p=generalized-parametric-contrastive-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalized-parametric-contrastive-learning/long-tail-learning-on-imagenet-lt)](https://paperswithcode.com/sota/long-tail-learning-on-imagenet-lt?p=generalized-parametric-contrastive-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalized-parametric-contrastive-learning/domain-generalization-on-imagenet-c)](https://paperswithcode.com/sota/domain-generalization-on-imagenet-c?p=generalized-parametric-contrastive-learning)



## Full ImageNet Classification and Out-of-Distribution Robustness
| Method | Model | Full ImageNet | ImageNet-C (mCE) | ImageNet-C (rel. mCE) | ImageNet-R | ImageNet-S | link | log | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPaCo      | ResNet-50   | 79.7 | 50.9 | 64.4 | 41.1 | 30.9 | [download](https://drive.google.com/file/d/1aO_Vo8v0ZRfDFLayVndkO3HJC6ZqIhKh/view?usp=sharing) | [download](https://drive.google.com/file/d/1jCLs6aPYAZW14A2YGij2tEJYEG5HCFmT/view?usp=sharing) |
| CE | ViT-B       | 83.6 | 39.1  | 49.9  | 49.9  | 36.1  | ---          | [download](https://drive.google.com/file/d/1Jv3R_kZL94jblzPgRhKNWTV56b9pa89l/view?usp=sharing) |
| CE | ViT-L       | 85.7 | 32.4  | 41.4  | 60.3  | 45.5  | ---          | [download](https://drive.google.com/file/d/1gfiKm7G27gHALCZUnVvaIRIsTd0DbMba/view?usp=sharing) |
| multi-task | ViT-B       | 83.4 | ---  | ---  | ---  | ---  | ---      | [download](https://drive.google.com/file/d/17CNEP8tIclaqwM9L_h6NXvgtujr7kNJT/view?usp=sharing)          |
| GPaCo      | ViT-B       | **84.0** | **37.2** | **47.3** | **51.7** | **39.4** | [download](https://drive.google.com/file/d/1DLZLXt7PH4NiJ5AqebA9f9iHgMiS8muz/view?usp=sharing) | [download](https://drive.google.com/file/d/1_0INOI53iq0PihO2-bF43cOKj_t9a847/view?usp=sharing) |
| GPaCo      | ViT-L       | **86.0** | **30.7** | **39.0** | **60.3** | **48.3** | [download](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155116769_link_cuhk_edu_hk/EYSTVmFZ99FEuOxt7LxiFesBoK2LNw2xC1H615Ir6Gd3ag?e=3ngaR0) | [download](https://drive.google.com/file/d/16NG5pOW2vMgSR0_SMYa2dMl00yr8tksP/view?usp=sharing) |

### CIFAR Classification
| Method | Model | Top-1 Acc(%) | link | log | 
| :---: | :---: | :---: | :---: | :---: |
| multi-task  | ResNet-50   | 79.1 | --- | [download](https://drive.google.com/file/d/11_oGhqTdgKXaoJtn4CQDQS1W1Li_UDpz/view?usp=sharing) |
| GPaCo       | ResNet-50   | 80.3 | --- | [download](https://drive.google.com/file/d/1olYyn4SQcwMEnQag1Q4lNpPnl45fuMbP/view?usp=sharing) |


## Long-tailed Recognition
### ImageNet-LT
 | Method | Model | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: |
 | GPaCo  | ResNet-50   | 58.5 | [download](https://drive.google.com/file/d/146_KqMYAeefAyavFfrysKa_jlnT3-DrC/view?usp=sharing) | [download](https://drive.google.com/file/d/1VFRbyLMrP8Oq_KeC-4QamcSh27u3cwG-/view?usp=sharing) |
 | GPaCo  | ResNeXt-50  | 58.9 | [download](https://drive.google.com/file/d/1_iTsXp2VJJ1zCOQL2eCjoAjFDEmcbpfE/view?usp=sharing) | [download](https://drive.google.com/file/d/1Zu5iAVhq4DJeORGlQvKtua3_rkRix8hB/view?usp=sharing) |
 | GPaCo  | ResNeXt-101 | 60.8 | [download](https://drive.google.com/file/d/1PkcgIgeMimb6zPSaOSAmTMsG8ahm3di1/view?usp=sharing) | [download](https://drive.google.com/file/d/1Up6pJBmi4vEJUgPNyGaiIJYWL3gMRZXG/view?usp=sharing) |
 | GPaCo  | ensemble( 2-ResNeXt-101) | 63.2 | --- | --- |
 
 ### iNaturalist 2018
 | Method | Model | Top-1 Acc(%) | link |  log |
 | :---: | :---: | :---: | :---: | :---: |
 | GPaCo  | ResNet-50   | 75.4 | [download](https://drive.google.com/file/d/1u2e0tvmHtgdcats5xgcAK76nes8SzjMW/view?usp=sharing) | [download](https://drive.google.com/file/d/1GvppGk4aRijqVhkcgRMqRz-niV4DQop9/view?usp=sharing) |
 | GPaCo  | ResNet-152  | 78.1 | --- | [download](https://drive.google.com/file/d/1eLO0ptdNLpM8Y0kaL9zFnS-rdlcb6PDS/view?usp=sharing) |
 | GPaCo  | ensembel(2-ResNet-152) | 79.8 | --- | --- |
 
 ### Places-LT
  | Method | Model | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: |
 | GPaCo  | ResNet-152   | 41.7 | [download](https://drive.google.com/file/d/1vbdM3ouFkt3qBWOMq59v1C8B8t0j2Nvc/view?usp=sharing) | [download](https://drive.google.com/file/d/1szk6kMElFnysZqbVS-ONkIIMS7fNuweo/view?usp=sharing) |
 
 ## Semantic Segmentation
 | Method | Dataset | Model | mIoU (s.s.) | mIoU (m.s.) | link | log | 
 | :---:  | :---: | :---: | :---: | :---: | :---: | :---: |
 | GPaCo  | ADE20K | Swin-T    | 45.4 | 46.8 | --- | [download](https://drive.google.com/file/d/1-ZC7r_SELvm9BnEPZsBu8d0mN8_7ltVZ/view?usp=sharing) |
 | GPaCo  | ADE20K | Swin-B    | 51.6 | 53.2 | --- | [download](https://drive.google.com/file/d/1N9fI2JeCp_Rq_RnoMKiRdd1q2lU0NxYy/view?usp=sharing) |
 | GPaCo  | ADE20K | Swin-L    | 52.8 | 54.3 | --- | [download](https://drive.google.com/file/d/1te4ejaiwbne2GGn0aYwlPUMHGlh34luZ/view?usp=sharing) |
 | GPaCo  | COCO-Stuff | ResNet-50    | 37.0 | 37.9 | --- | [download](https://drive.google.com/file/d/1ox_-EbwbfBwNnb9_dYnQeCtyJoXQ9KVn/view?usp=sharing) |
 | GPaCo  | COCO-Stuff | ResNet-101   | 38.8 | 40.1 | --- | [download](https://drive.google.com/file/d/1OIXkxu5xo3_0crKlpgHhZMofPKLVXpVC/view?usp=sharing) |
 | GPaCo  | Pascal Context 59 | ResNet-50    | 51.9 | 53.7 | --- | [download](https://drive.google.com/file/d/1_aVw-_ThzPNtlcpAarJNOPYuPZ04QDxO/view?usp=sharing) |
 | GPaCo  | Pascal Context 59 | ResNet-101   | 54.2 | 56.3 | --- | [download](https://drive.google.com/file/d/1PAA1v_BnGtGXXZCbKJ1BzsQg-SKZl63i/view?usp=sharing) |
 | GPaCo  | Cityscapes | ResNet-18    | 78.1 | 79.7 | --- | [download](https://drive.google.com/file/d/1AxPqf3MRbAODefw0njpwmlEii7z7eyuc/view?usp=sharing) |
 | GPaCo  | Cityscapes | ResNet-50    | 80.8 | 82.0 | --- | [download](https://drive.google.com/file/d/1Hio9KIc14zyBaCBlXE0j7fM7VdEhA7BB/view?usp=sharing) |
 | GPaCo  | Cityscapes | ResNet-101   | 81.4 | 82.1 | --- | [download](https://drive.google.com/file/d/1Ph9ijtFLrfLD9rax1KNfvwW0bXewMiO7/view?usp=sharing) |
 
 ## Get Started
 ### Environments
 We use python3.8, pytorch 1.8.1, mmcv 1.3.13 and timm==0.3.2. Our code is based on [PaCo](https://github.com/dvlab-research/parametric-contrastive-learning), [MAE](https://github.com/facebookresearch/mae), and [mmseg](https://github.com/open-mmlab/mmsegmentation).
 
 ### Train and Evaluation Scripts
 #### On full ImageNet and OOD robustness,
 We use 8 Nvidia GForce RTx 3090 GPUs. MAE pretrained models should be downloaded from [here](https://github.com/facebookresearch/mae). 
 ```
 cd GPaCo/LT
 bash sh/ImageNet/train_resnet50.sh
 bash sh/ImageNet/eval_resnet50.sh

 cd GPaCo/MAE-ViTs
 bash sh/finetune_base_mae.sh
 bash sh/finetune_base_mae_multitask.sh
 bash sh/finetune_base_mae_gpaco.sh
 bash sh/finetune_base_mae_gpaco_eval.sh
 ```
 
 #### On imbalanced data,
 ```
 cd GPaCo/LT
 bash sh/LT/ImageNetLT_train_X50_multitask.sh
 bash sh/LT/ImageNetLT_train_X50.sh
 sh/LT/ImageNetLT_eval_X50.sh
 
 bash sh/LT/Inat_train_R50.sh
 sh/LT/Inat_eval_R50.sh
 
 bash sh/LT/PlacesLT_train_R152.sh
 bash sh/LT/PlacesLT_eval_R152.sh
 ```
 
 #### On semantic segmentation,
 ```
 cd GPaCo/Seg/semseg
 bash sh/ablation_paco_ade20k/upernet_swinbase_160k_ade20k_paco.sh
 bash sh/ablation_paco_coco10k/r50_deeplabv3plus_40k_coco10k_paco.sh
 bash sh/ablation_paco_context/r50_deeplabv3plus_40k_context_paco.sh
 bash sh/ablation_paco_cityscapes/r50_deeplabv3plus_40k_context.sh
 ```
 
 







# Parametric-Contrastive-Learning
This repository contains the implementation code for ICCV2021 paper:  
**Parametric Contrastive Learning** (https://arxiv.org/abs/2107.12028)     


## Overview

In this paper, we propose Parametric Contrastive Learning (PaCo) to tackle long-tailed recognition. Based on theoretical analysis, we observe supervised contrastive loss tends to bias on high-frequency classes and thus increases the difficulty of imbalance learning. We introduce a set of parametric class-wise learnable centers to rebalance from an optimization perspective. Further, we analyze our PaCo loss under a balanced setting. Our analysis demonstrates that PaCo can adaptively enhance the intensity of pushing samples of the same class close as more samples are pulled together with their corresponding centers and benefit hard example learning. **Experiments on long-tailed CIFAR, ImageNet, Places, and iNaturalist 2018 manifest the new state-of-the-art for long-tailed recognition. On full ImageNet, models trained with PaCo loss surpass supervised contrastive learning across various ResNet backbones**.


## Results and Pretrained models
### Full ImageNet (Balanced setting)

 | Method | Model | Top-1 Acc(%) | link | log |  
 | :---: | :---: | :---: | :---: | :---: |
 | PaCo  | ResNet-50  | 79.3 | [download](https://drive.google.com/file/d/1fBbTWJlM3knjN0SIjrLhQw_TeUuoHsOe/view?usp=sharing) | [download](https://drive.google.com/file/d/1G_oTmBxAxwZdIa4YJSCJ-lYslwvsBMiF/view?usp=sharing) |
 | PaCo  | ResNet-101 | 80.9 | [download](https://drive.google.com/file/d/18lAQu33eN4pzhOi6lSmvfK6ORvD0gMGg/view?usp=sharing) | [download](https://drive.google.com/file/d/1axS5ryB-MjoKBRN4N9lVHyLQ2pOfKgMw/view?usp=sharing) |
 | PaCo  | ResNet-200 | 81.8 | [download](https://drive.google.com/file/d/14ZOI8tdUMGZFp08QfYog-aRr541psmur/view?usp=sharing) | [download](https://drive.google.com/file/d/18RNyK9HHSeQOQj69YUX_5Seq_PdGWavv/view?usp=sharing) | 
 
### ImageNet-LT (Imbalance setting)
 | Method | Model | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: |
 | PaCo  | ResNet-50   | 57.0 | [download](https://drive.google.com/file/d/1a73Ez_k47S2hmD_0L-sLH0OEhxK8SpQt/view?usp=sharing) | [download](https://drive.google.com/file/d/1NmynVzdkSye0FNEEyHSFF1oMc6q2wypJ/view?usp=sharing) |
 | PaCo  | ResNeXt-50  | 58.2 | [download](https://drive.google.com/file/d/1J7pvp-CWx7e2hPFNa1a05Oy9igHSe1eM/view?usp=sharing) | [download](https://drive.google.com/file/d/1ssvsA-xG2oj5wUwmC-Gu_pVZdg9POp7R/view?usp=sharing) |
 | PaCo  | ResNeXt-101 | 60.0 | [download](https://drive.google.com/file/d/1k14zhOwF8NBTb17mUN_UAGBkIIZsVBCV/view?usp=sharing) | [download](https://drive.google.com/file/d/1ZVwUKFb9AozaNKb8aSUXLCy27LgE7Kt2/view?usp=sharing) |
 
 ### iNaturalist 2018 (Imbalanced setting)
 | Method | Model | Top-1 Acc(%) | link |  log |
 | :---: | :---: | :---: | :---: | :---: |
 | PaCo  | ResNet-50   | 73.2 | TBD | [download](https://drive.google.com/file/d/1oYMqMcE9uC1pXwEOapB7zoha6Pjj7WO4/view?usp=sharing) |
 | PaCo  | ResNet-152  | 75.2 | TBD | [download](https://drive.google.com/file/d/1i5g10hlgNiPWOZ1zHn0wAhNag5F1ak9F/view?usp=sharing) |
 
 ### Places-LT (Imbalanced setting)
  | Method | Model | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: |
 | PaCo  | ResNet-152   | 41.2 | TBD | [download](https://drive.google.com/file/d/1kwu8AB5slPZLRm3OI-k3Jd6p22yMOThW/view?usp=sharing) |
 
## Get Started
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

# BibTex
If you find this code or idea useful, please consider citing our work:
```
@ARTICLE{10130611,
  author={Cui, Jiequan and Zhong, Zhisheng and Tian, Zhuotao and Liu, Shu and Yu, Bei and Jia, Jiaya},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Generalized Parametric Contrastive Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TPAMI.2023.3278694}}


@inproceedings{cui2021parametric,
  title={Parametric contrastive learning},
  author={Cui, Jiequan and Zhong, Zhisheng and Liu, Shu and Yu, Bei and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={715--724},
  year={2021}
}

@ARTICLE{9774921,
  author={Cui, Jiequan and Liu, Shu and Tian, Zhuotao and Zhong, Zhisheng and Jia, Jiaya},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={ResLT: Residual Learning for Long-Tailed Recognition}, 
  year={2023},
  volume={45},
  number={3},
  pages={3695-3706},
  doi={10.1109/TPAMI.2022.3174892}
  }

  
@article{cui2022region,
  title={Region Rebalance for Long-Tailed Semantic Segmentation},
  author={Cui, Jiequan and Yuan, Yuhui and Zhong, Zhisheng and Tian, Zhuotao and Hu, Han and Lin, Stephen and Jia, Jiaya},
  journal={arXiv preprint arXiv:2204.01969},
  year={2022}
  }
  
@article{zhong2023understanding,
  title={Understanding Imbalanced Semantic Segmentation Through Neural Collapse},
  author={Zhong, Zhisheng and Cui, Jiequan and Yang, Yibo and Wu, Xiaoyang and Qi, Xiaojuan and Zhang, Xiangyu and Jia, Jiaya},
  journal={arXiv preprint arXiv:2301.01100},
  year={2023}
}
```
 
 
 
 

