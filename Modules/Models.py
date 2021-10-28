'''
이 파일에 학습될 모델을 정의합니다.
'''
import segmentation_models_pytorch as smp

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