# Residual Distillation: Towards Portable Deep Neural Networks without Shortcuts

**Implementation for our paper: Revisiting Knowledge Distillation via Label Smoothing Regularization, [arxiv]() (we will release our paper soon).**


Residual neural networks (ResNets) have delivered state-of-the-art performance on many vision problems. However, the computationally intensive nature of ResNets make them difficult to deploy on embedded systems with limited resources.  

The shortcuts in ResNets accounts for around 40 percent of the total feature map data that consumes much off-chip memory traffic, resulting in a major performance bottleneck. In this work, we consider how the shortcuts can be removed to improve deployment efficiency of ResNets  without hurting the model accuracy. In particular, we propose a novel joint-training framework to facilitate the training of plain CNN without shortcuts by using the gradient of the ResNet counterpart as well as knowledge distillation from internal features. In this framework, early stages of plain CNN connects to both later stages of itself and later stages of ResNets. Specifically, during backpropagation, the gradients are calculated as a mixture of these two parts.  This framework allows us to benefit from the shortcut during the training phase and to deploy the model without shortcuts.  Experiment on ImageNet shows that by using the proposed framework, a 50-layer plain CNN model can achieve the same level of accuracy as ResNet50 with up to 1.4 times speedup and 20 percent memory reduction. 

![](/figures/main_arch.png)
## 1. Preparations:

Clone this repository:

```
git clone https://github.com/leoozy/JointRD_Neurips2020.git
```

### 1.1 Environment
Better use: NVIDIA GPU + CUDA 10.0 + Pytorch 1.3.1

### 1.2 Dataset

We use the public image classification datasest: [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet](http://www.image-net.org/); You can download them to "/cache/dataset/". You can also change the varaible "data_dir" in all *.sh files.

## 2. Train baseline models

You can skip this step by using pytorch-pretrained models in [here](https://pytorch.org/docs/stable/torchvision/models.html). Download to the pre-trained model to "--tmodel_name" dir.

You can also train  the pre-trained model by yourself.

For example, normally train ResNet18 to obtain the ImageNet pre-trainede teacher:
```
bash ./script/resnet_imagenet.sh 18 0.0 RES_NMT 1e-4  
```
the first parameter (18) can be chage to 34 or 50 to specify the layer of the resnet model.

Normally train ResNet18 to obtain the CIFAR100 pre-trainede teacher:
```
bash ./script/resnet_icifar.sh 18 0.0 RES_NMT 
```
you can change the dataset varaible in resnet_icifar.sh to specify the used dataset.

### 3. Exploratory experiments (Section Experiments in our paper)

### 3.1 Training the plain-CNN baseline


Normally train plain-CNN model to obtain the plain-CNN baseline on the ImageNet dataset:

```
bash ./script/imagenet_plaincnn.sh num_of_layers 0.0 CNN_NMT 
```
Normally train plain-CNN model to obtain the plain-CNN baseline on the CIFAR dataset:
```
bash ./script/CIFAR_plaincnn.sh num_of_layers 0.0 CNN_NMT 
```

### 3.2 Training the JointRD model

Normally train the JointRD model on the ImageNet dataset:
```
bash ./script/resnet_imagenet.sh num_of_layers 5e-3 TA 1e-4`
```

Normally train the JointRD model on the CIFAR dataset:
```
bash ./script/CIFAR_plaincnn.sh num_of_layers 0.5 TA seed
```
