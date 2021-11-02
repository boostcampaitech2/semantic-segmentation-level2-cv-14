'''
이 파일에 TTA를 위한 Transform을 정의합니다.
'''

import ttach as tta

def Default():
    return tta.Compose(
        [
        ]
    )

def HorizontalFlip():
    return tta.Compose(
        [
            tta.HorizontalFlip(),
        ]
    )

def HorizontalFlip_Rotate90():
    return tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
        ]
    )

def HorizontalFlip_Rotate90_MultiScale():
    return tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
            tta.Scale(scales=[1, 0.5, 2]),
        ]
    )

def HorizontalFlip_Rotate90_MultiScale_Multiply():
    return tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
            tta.Scale(scales=[1, 2, 4]),
            tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )