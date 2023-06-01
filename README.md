
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
|    <br> Do not use DDA   |    <br>Resnet34<br>Resnet50 |    <br>86.70%<br>86.88% |
|      <br> Stage 1,2      |    <br>Resnet34<br>Resnet50 |    <br>86.66%<br>86.76% |
|     <br> Stage 1,2,3     |    <br>Resnet34<br>Resnet50 |    <br>86.57%<br>86.44% |
|         <br>Stage 2,3,4  |    <br>Resnet34<br>Resnet50 |    <br>87.12%<br>86.73% |
|       <br>Stage 3,4      |    <br>Resnet34<br>Resnet50 |    <br>87.35%<br>87.32% |
|       <br>All stage      |    <br>Resnet34<br>Resnet50 |    <br>87.50%<br>87.61% |
