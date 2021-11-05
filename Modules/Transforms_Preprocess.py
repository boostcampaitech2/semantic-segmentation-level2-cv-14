'''
이 파일에 이미지 전처리를 위한 Transform을 정의합니다.
'''

import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2

def NoTransform():
    return alb.Compose([])

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

def HorizontalFlip_Rotate90_RandomResizedCrop():
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.RandomResizedCrop(height=512,width=512, scale=[0.5, 1], p=0.5),
        alb.Normalize(),
        ToTensorV2()
    ])

def HorizontalFlip_Rotate90_Elastic():
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.ElasticTransform(p=0.5, alpha=40, sigma=40 * 0.05, alpha_affine=40 * 0.03),
        alb.Normalize(),
        ToTensorV2()
    ])

def HorizontalFlip_Rotate90_CLAHE():
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.CLAHE(p=0.5),
        alb.Normalize(),
        ToTensorV2()
    ])


def HorizontalFlip_Rotate90_GridMask():
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.Normalize(),
        alb.GridDropout(
            holes_number_x=5,holes_number_y=5,
            ratio=0.3, p=1.0),
        ToTensorV2()
    ])


def HorizontalFlip_Rotate90_CLAHE_GridMask():
    return alb.Compose(
        [
            alb.HorizontalFlip(p=0.5),
            alb.RandomRotate90(p=0.5),
            alb.CLAHE(p=0.5, clip_limit=2, tile_grid_size=(20, 20)),
            alb.GridDropout(holes_number_x=5,holes_number_y=5,ratio=0.3, p=0.5),
            alb.Normalize(),
            ToTensorV2(),
        ]
    )
    

def ObjectAugmentation():
    return alb.Compose([
        alb.ShiftScaleRotate((-0.5,0.5),(-0.5,0.5),(-45,45), border_mode=cv2.BORDER_CONSTANT),
        alb.HorizontalFlip(p=0.5)
    ])