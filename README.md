## Resnet50 fixing for rafdb
kết quả 1

kết quả 2

kết quả 3

SEED 113 with zoom got 74.5%

- thực nghiệm 1
```python

seg = iaa.Sometimes(
        0.5,
	iaa.Sequential([iaa.Fliplr(p=0.7),iaa.Affine(rotate=(-30, 30))]),
        iaa.Sequential([iaa.Affine(scale=(1.0, 1.1))]),
# 	iaa.Sequential([iaa.Fliplr(p=1),iaa.Affine(rotate=(-30, 30))]),
#         iaa.Sequential([iaa.Affine(scale=(1.1))]),

```
- thực nghiệm 2
```python
seg = iaa.Sometimes(
        0.5,
	iaa.Sequential([iaa.Fliplr(p=0.7),iaa.Affine(rotate=(-30, 30))]),
        iaa.Sequential([iaa.Affine(scale=(1.1))]),
# 	iaa.Sequential([iaa.Fliplr(p=1),iaa.Affine(rotate=(-30, 30))]),
#         iaa.Sequential([iaa.Affine(scale=(1.1))]),
```
