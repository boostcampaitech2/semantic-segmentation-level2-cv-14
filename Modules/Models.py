'''
이 파일에 학습될 모델을 정의합니다.
'''
import segmentation_models_pytorch as smp
import yaml
import torch.nn.functional as F
import torch
import torch.nn as nn

import sys
import os
module_path = os.path.join(os.getcwd(), 'Modules') 
sys.path.append(module_path)

from Hrnet_Sources.Model.seg_hrnet_ocr import get_seg_model
from Hrnet_Sources.Model.seg_hrnet import get_seg_model

def UNetPP_Efficientb2():
    return smp.UnetPlusPlus(
        encoder_name="efficientnet-b2",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=11,  # model output channels (number of classes in your dataset)
    )

def UNetPP_Efficientb4():
    return smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=11,  # model output channels (number of classes in your dataset)
    )

def DeepLabV3P_Efficientb4():
    return smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=11,  # model output channels (number of classes in your dataset)
    )

class Hrnet_Seg_Ocr_Model(nn.Module):
    def __init__(self):
        super().__init__()
        config_path = 'Modules/Hrnet_Sources/Config/hrnet_seg.yaml'
        with open(config_path) as f:
            cfg = yaml.load(f,Loader=yaml.FullLoader)
        self.encoder = get_seg_model(cfg)

    def forward(self, x):
        x = self.encoder(x)
        x = F.interpolate(input=x, size=(512, 512), mode = 'bilinear', align_corners=True)
        return x

def Hrnet_Seg_Ocr():    
    return Hrnet_Seg_Ocr_Model()