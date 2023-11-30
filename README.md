
# Dual-Domain-Attention in Facial Expression Recoginition
## Summary:
The code for Dual-Domain Attention, this is one of my proposal in my graduation thesis for Facial Expression Recoginition


![Net](https://github.com/Harly-1506/Dual-Domain-Attention/assets/86733695/d1755a67-c51d-4ef8-815b-20e6388277db)

## How to train?
```python
git clone https://github.com/Harly-1506/Dual-Domain-Attention.git
%cd Dual-Domain-Attention
#Choose your model and run:
python main_fer2013.py #main_rafdb.py
```

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

