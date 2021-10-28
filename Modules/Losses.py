'''
이 파일에 학습에 사용될 Loss를 정의합니다.
'''
import torch.nn as nn

def CrossEntropyLoss():
    return nn.CrossEntropyLoss()