
# Dual-Domain-Attention in Facial Expression Recognition
![Static Badge](https://img.shields.io/badge/Pyhon-3.10.10-blue)
![Static Badge](https://img.shields.io/badge/Framework-Pytorch-Green)

# Note: 
The ideas in my thesis have been shared with another person and have been published at the 2023 ICIP conference and ASK conference, but I am not listed as an author (Even though it was my idea - proposed 1 and proposed 2). So, if you read a paper or a thesis with similar ideas, please note that I'm not the one who copied them :)

## Summary:
The code for Dual-Domain Attention, this is one of my proposal in my graduation thesis for Facial Expression Recoginition, read for my proposal more [here](https://harly.vercel.app/graduation-thesis-facial-emotion-recognition-deep-learning-application-combines-attention)


![Net](https://github.com/Harly-1506/Dual-Domain-Attention/assets/86733695/d1755a67-c51d-4ef8-815b-20e6388277db)

## Datasets
- [FER2013](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/overview)
- [RAF-DB](http://www.whdeng.cn/raf/model1.html)
## How to train?
- To train the model, you need to adjust the parameters in the configs file of each dataset if you want, then select the model in the main file of each dataset then run it. Make sure you use the correct library in requirements.txt 
```python3
git clone https://github.com/Harly-1506/Dual-Domain-Attention.git
cd Dual-Domain-Attention
#Choose your model and run:
python main_fer2013.py #main_rafdb.py
```
## Evaluation
```python3
python cm_rafdb.py
```
![CM_ResNet50_Vggface + DDA-1](https://github.com/Harly-1506/Dual-Domain-Attention/assets/86733695/9163ea8f-c16f-45b7-a896-d33b31f5f9cb)

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

## Authors
- [Harly-1506](github.com/Harly-1596)
## References
- In thesis
- Thanks for great source code [Luan-Pham](https://github.com/phamquiluan)

