from imgaug import augmenters as iaa
import albumentations as A

#
seg_fer = iaa.Sometimes(
        0.5,
	iaa.Sequential([iaa.Fliplr(p=0.6),iaa.Affine(rotate=(-30, 30))]),
        iaa.Sequential([iaa.Affine(scale=(1.0, 1.1))]),

)
seg_fertest2 = iaa.Sequential([iaa.Affine(scale=(1.0, 1.1))])
seg_fertest1 = iaa.Sequential([iaa.Fliplr(p=0.5),iaa.Affine(rotate=(-30, 30))])


#augmentaion for Image-net
seg_raf = iaa.Sometimes(
        0.5,
	iaa.Sequential([iaa.Fliplr(p=0.5), iaa.Affine(rotate=(-25, 25))]),
        iaa.Sequential([iaa.RemoveSaturation(1),iaa.Affine(scale=(1.0, 1.05)) ])
)
seg_raftest2 = iaa.Sequential([iaa.RemoveSaturation(1),iaa.Affine(scale=(1.0, 1.05))])
seg_raftest1 = iaa.Sequential([iaa.Fliplr(p=0.5), iaa.Affine(rotate=(-25, 25))])
