import albumentations as A
from albumentations.pytorch import ToTensorV2

'''
train_transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.RandomRotate90(p=0.5),
                            #A.OneOf([
                            #    A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                            #    A.GridDistortion(p=1),
                            #    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1)
                            #], p=0.6),
                            ToTensorV2()
                            ])
'''
train_transform = A.Compose([
                          ToTensorV2()
                          ])
val_transform = A.Compose([
                          ToTensorV2()
                          ])

test_transform = A.Compose([
                           ToTensorV2()
                           ])

#오류: albumentation from .cv2 import * ImportError: libSM.so.6: cannot open shared object file: No such file or directory  
#해결: sudo apt-get install libsm6 libxrender1 libfontconfig1
