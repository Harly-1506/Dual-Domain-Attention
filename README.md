
# Dual-Domain-Attention in Facial Expression Recoginition
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

## Using DDA for emotion in context (EMOTIC dataset)
- Reuse the trained DDA on the RAF-DB dataset to make a face feature extraction model extracted from the annotation of the EMOTIC dataset

![Emotic_model](https://github.com/Harly-1506/Dual-Domain-Attention/assets/86733695/9a602c2e-2fd1-414e-9f06-1185c7919c5a)

- Result: only using body for training.

|     Model 1         |     Model 2         |     Model 3     |     mAP     |
|---------------------|---------------------|-----------------|-------------|
|     Resnet50        |     Resnet34-DDA    |     Resnet34    |     29.8    |
|     Resnet50        |     Resnet34-DDA    |     X           |     29.4    |
|     Resnet50-DDA    |     Resnet18        |     Resnet34    |     30.3    |
|     Resnet50-DDA    |     Resnet18        |     X           |     29.6    |
|     Resnet50-DDA    |     Resnet34-DDA    |     X           |     30.6    |
|     Resnet50-DDA    |     Resnet34-DDA    |     Resnet34    |     31.7    |

- Compare with Sota Methods just using only body annotation

| Emotions Categories                                                                                                                                                                                                                                                                                                                         | Kositi                                                                                                                                                                                                                                 | Hoang                                                                                                                                                                                                                                  | Mittal                                                                                                                                                                                                                                 | Ours                                                                                                                                                                                                                                   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Affection<br>Anger<br>Annoyance<br>Anticipation<br>Aversion<br>Confidence<br>Disapproval<br>Disconnection<br>Disquietment<br>Doubt/Confusion<br>Embarrassment<br>Engagement<br>Esteem<br>Excitement<br>Fatigue<br>Fear<br>Happiness<br>Pain<br>Peace<br>Pleasure<br>Sadness<br>Sensitivity<br>Suffering<br>Surprise<br>Sympathy<br>Yearning | 27.85<br>09.49<br>14.06<br>58.64<br>07.48<br>78.35<br>14.97<br>21.32<br>16.89<br>29.63<br>03.18<br>87.53<br>17.73<br>77.16<br>09.70<br>14.14<br>58.26<br>08.94<br>21.56<br>45.46<br>19.66<br>09.28<br>18.84<br>18.81<br>14.71<br>08.34 | 37.07<br>18.67<br>20.74<br>57.96<br>10.81<br>76.81<br>19.65<br>30.16<br>19.48<br>21.76<br>02.65<br>87.47<br>15.25<br>72.49<br>16.38<br>05.95<br>79.99<br>12.19<br>24.68<br>50.05<br>30.46<br>06.87<br>31.18<br>14.11<br>12.81<br>09.57 | 29.87<br>08.52<br>09.65<br>46.23<br>06.27<br>51.92<br>11.81<br>31.74<br>07.57<br>21.62<br>08.43<br>78.68<br>18.32<br>73.19<br>06.34<br>14.29<br>52.52<br>05.75<br>13.53<br>58.26<br>19.94<br>03.16<br>15.38<br>05.29<br>22.38<br>04.94 | 27.55<br>23.61<br>24.25<br>59.22<br>12.83<br>82.40<br>19.68<br>32.43<br>23.90<br>21.37<br>02.00<br>86.62<br>15.92<br>73.56<br>14.89<br>06.71<br>81.11<br>12.29<br>23.37<br>53.61<br>36.92<br>10.71<br>40.40<br>14.30<br>14.98<br>10.09 |
| mAP                                                                                                                                                                                                                                                                                                                                         | 27.38                                                                                                                                                                                                                                  | 30.20                                                                                                                                                                                                                                  | 24.06                                                                                                                                                                                                                                  | 31.72                                                                                                                                                                                                                                  
______________________________________________________________________________
## Conclusion
- All of the above methods are included in my graduate thesis
- Coding and training made by me and re-trainable
