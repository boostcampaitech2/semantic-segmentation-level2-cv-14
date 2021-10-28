'''
이 파일에 이미지 전처리를 위한 Transform을 정의합니다.
'''

import albumentations as alb
from albumentations.pytorch import ToTensorV2

def Default():
    return alb.Compose([
        alb.Normalize(),
        ToTensorV2()
    ])

def HorizontalFlip():
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.Normalize(),
        ToTensorV2()
    ])

def HorizontalFlip_Rotate90():
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.Normalize(),
        ToTensorV2()
    ])

def HorizontalFlip_Rotate90_Multiscale():
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.Normalize(),
        ToTensorV2()
    ])