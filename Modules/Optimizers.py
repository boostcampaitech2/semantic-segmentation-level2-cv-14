'''
이 파일에 학습에 사용될 Optimizer를 정의합니다.
'''

import torch

def Adam(model, lr, weight_decay):
    return torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

def AdamW(model, lr, weight_decay):
    return torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)

def SGD(model, lr, weight_decay):
    return torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay)

