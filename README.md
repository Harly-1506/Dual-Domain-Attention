
# Dual-Domain-attention in Facial Expression Recoginition
## Summary:
The code for Dual-Domain Attention ICIP2023 (submitting)

![Net](https://github.com/Harly-1506/Dual-Domain-Attention/assets/86733695/08522388-9483-4f61-9f61-586a514261d1)

**Abstract**: Attention mechanisms have become crucial in contemporary techniques for recognizing emotions through facial expressions. In this work, we proposed a novel dual-domain attention module incorporating local context in the spatial domain and global context in the context domain of feature maps. Our attention module is to learn residual attention
from dual-domain feature maps for improving intermediate
feature maps to focus on the critical parts of a face, such as
the eyes, nose, and mouth, which are known to carry important information about emotions. By doing so, the model can
generate more meaningful representations of facial features
and improve accuracy in emotion recognition tasks. Experiments on FER2013 and RAF-DB have demonstrated superior
performance compared to existing state-of-the-art methods.

## Experiments:
These DDA blocks are tested when attached to Resnet networks and the results are shown in the table below
|     Models            |     Pre-trained    |     FER2013 (%)    |     RAF-DB (%)    |        
|-----------------------|--------------------|--------------------|-------------------|
|     Resnet34          |     Image-Net      |     72.80%         |     86.70%        |
|     Resnet50          |     Image-Net      |     73.40%         |     86.99%        |
|     Resnet34 + DDA    |     Image-Net      |     74.75%         |     87.50%        |
|     Resnet50 + DDA    |     Image-Net      |     73.72%         |     87.61%        |
|     Resnet50          |     VGGface2       |     74.30%         |     88.90%        |
|     Resnet50 + DDA    |     VGGface2       |     74.67%         |     89.96%        |

## Alation study:

|     <br>DDA in Stages    |         <br>Backbone        |      <br>RAF-DB (%)     |
|:------------------------:|:---------------------------:|:-----------------------:|
|     Do not use DDA   |    Resnet34<br>Resnet50 |    86.70%<br>86.88% |
|     Stage 1,2      |    Resnet34<br>Resnet50 |    86.66%<br>86.76% |
|     Stage 1,2,3     |   Resnet34<br>Resnet50 |    86.57%<br>86.44% |
|     Stage 2,3,4  |    Resnet34<br>Resnet50 |    87.12%<br>86.73% |
|     Stage 3,4      |    Resnet34<br>Resnet50 |  87.35%<br>87.32% |
|     All stage      |    Resnet34<br>Resnet50 |  87.50%<br>87.61% |

This table can be
concluded that the different stages play a complementary role
in creating the best possible feature maps through the training
process. The optimal results are obtained when all stages are
used together with DDA

## Comparisons with Sota Methods:
We benchmark our code thoroughly on two datasets: FER2013 and RAF-DB
| Sota                | FER2013(%)       | Sota                             | RAF-DB(%)         |
|---------------------|------------------|----------------------------------|-------------------|
| Inception           |    71.60%    | RAN                              |    86.90%     |
| MLCNNs              |    73.03%    | SCN                              |    87.03%     |
| Resnet50 + CBam     |    73.39%    | DACL                             |    87.78%     |
| ResMaskingNet       |    74.14%    | KTN                              |    88.07%     |
| LHC-Net             |    74.42%    | EfficientFace                    |    88.36%     |
| **Resnet50+DDA (ours)** |    **74.67%**    | DAN                              |    89.70%     |
| **Resnet34+DDA (ours)** |    **74.75%**    |   **ResNet-50 + DDA (ours)**    |    **89.96%**     |

## Fusion Attention method with DDA:
- [Fusion Attention method](https://1drv.ms/b/s!Avr_XL5_YnvQhTpXv7KfVXE1acnT?e=jRH9sS) described in this [paper](https://1drv.ms/b/s!Avr_XL5_YnvQhTpXv7KfVXE1acnT?e=jRH9sS)
- Results: Other Fusion methods reduce the accuracy when combining Resnet models using DDA. Our proposed method has the highest results so far on the RAF-DB dataset

|    <br>Fusion methods       |    <br>Model   1                                                                |    <br>Model   2                                                                      |    <br>RAF-DB   (%)                                          |
|-------------------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|--------------------------------------------------------------|
|    <br>Late   Fusion                |    Resnet18   <br>VGG11<br>VGG11<br>Resnet34<br>Resnet50+DDA (Image-net) |    Resnet34<br>VGG13<br>Resnet34<br>Resnet50+DDA<br>Resnet50+DDA (VGGface2)    |    86.35%<br>86.08%<br>86.08%<br>89.21%<br>89.66%<br>    |
|    <br>Early   Fusion               | Resnet18<br>VGG13<br>VGG11<br>Resnet34<br>Resnet50+DDA (Image-net)           | Resnet34<br>Resnet34<br>Resnet34<br>Resnet50+DDA<br>Resnet50+DDA (VGGface2)        | 86.66%<br>85.49%<br>86.08%<br>88.97%<br>89.65%               |
|    <br>Joint   fusion               |    Resnet18   <br>VGG13<br>VGG11<br>Resnet34<br>Resne50+DDA (Image-Net)  |    Resnet34<br>Resnet34<br>Resnet34<br>Resnet50+DDA<br>Resnet50+DDA (VGGface2) |    86.05%<br>86.63%<br>86.40%<br>89.70%<br>88.23%        |
|    **<br>Fusion   attention (ours)**    |    **Resnet18   <br>VGG13<br>Resnet34<br>Resnet50+DDA (Image-net)**          |    **Resnet34   <br>Resnet34<br>Resnet50+DDA<br>Resnet50+DDA (VGGface2)**          |    **90.95%   <br>90.92%<br>92.54%<br>93.38%**           |
