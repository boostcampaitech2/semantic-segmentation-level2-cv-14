'''
이 파일에 각종 편의기능을 정의합니다.
'''

import os
import torch
import numpy as np
import random

def Fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def CreateDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]