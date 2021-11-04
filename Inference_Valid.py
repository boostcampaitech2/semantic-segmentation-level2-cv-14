'''
이 코드는 다음 기능을 포함합니다.
 - pytorch 기반의 Semantic Segmentation Model 을 Inference 해서 submission.csv 파일을 생성합니다.
 - config.json 파일을 통해 추론 인자를 제어할 수 있습니다.

 - 사용법
    Inference_Test.py --config Configs/UNetPP_Efficientb4_aug_test.json

 - 출력
    'path_save' 값으로 지정한 경로에 다음과 같은 파일이 생성됩니다.
        submission.csv : 부스트캠프 컴피티션 형식에 맞추어 생성된 Segmentation 결과 파일입니다.

 - 종속 파일 명세
    - Modules/Data.py : Custom Dataset 클래스들이 정의되어 있습니다. (현재는 부스트캠프 컴피티션용 COCO format만 지원합니다.)
    - Modules/Models.py : Semantic Segmentation 모델들이 정의되어 있습니다.
    - Modules/Transform_AfterProcess.py : 이미지 후처리를 위한 Transform들이 정의되어 있습니다.
    - Modules/Transform_Preprocess.py : 이미지 전처리를 위한 Transform들이 정의되어 있습니다.
    - Modules/Transform_TTA.py : TTA를 위한 Transform 들이 정의되어 있습니다.
    - Utils/Tools.py : 각종 편의기능들이 정의되어 있습니다.

 - 기본 config_test.json 요소 명세
    - path_dataset_root : 데이터셋이 저장된 root를 정의합니다.
    - path_test_json : test 데이터 json 파일을 정의합니다.
    - path_checkpoint : 학습된 모델의 weight가 저장된 경로를 정의합니다.
    - random_fix : 난수 고정 여부를 정의합니다.
    - random_seed : 고정할 seed를 정의합니다.
    - model : Semantic Segmentation 모델을 정의합니다.
    - model_batch_size : 모델의 배치 크기를 정의합니다.
    - data_num_workers : data loader 가 사용할 프로세스 수를 정의합니다.
    - data_loading_mode : 데이터를 미리 로드하거나, 실시간으로 로드하는 모드를 정의합니다. 'preload' 또는 'realtime' 으로 선택할 수 있습니다.
    - data_test_transform : 테스트에 사용될 전처리 transform을 정의합니다. ('Default'를 추천합니다.)
    - data_tta_transform : TTA에 사용될 transform을 정의합니다.
    - data_target_size : 추론 후 submission을 생성할 때 변환돨 이미지 크기를 정의합니다. (256을 추천합니다.)

작성자 JiSeong Kim
최초 작성일 2021-10-28
'''

#Built-in
from multiprocessing import freeze_support, Pool
import argparse
import json
from itertools import repeat

# External
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import ttach as tta


# Custom
from Modules import Data, Models, Transforms_Preprocess, Transforms_TTA, Losses
from Modules.Transforms_AfterProcess import DCRF_SubRoutine
from Utils import Tools
from Utils.utils import label_accuracy_score, add_hist


def Main(args):
    # seed 고정
    if args['random_fix']:
        Tools.Fix(args['random_seed'])

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 모델 정의
    model = getattr(Models, args['model'])()

    # Weight 로드
    checkpoint = torch.load(args['test_path_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint.state_dict())

    # device 할당
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        args['test_model_batch_size'] = args['test_model_batch_size'] * torch.cuda.device_count()
    model = model.to(device)

    # TTA Wrapper 적용
    model = tta.SegmentationTTAWrapper(model, getattr(Transforms_TTA, args['test_data_transform_tta'])())

    # 데이터셋 정의
    test_dataset = Data.DataSet_Trash(args['path_json_valid'], args['path_dataset_root'], loading_mode=args['test_data_loading_mode'], transforms=getattr(Transforms_Preprocess, args['test_data_transform_preprocess'])(), stage='train')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args['test_model_batch_size'], shuffle=False, num_workers=args['test_data_num_workers'])

    # Loss function 정의
    criterion = getattr(Losses, args['train_loss'])()

    n_class = len(test_dataset.Categories())
    model.eval()
    with torch.no_grad():

        total_loss = 0
        hist = np.zeros((n_class, n_class))
        for image, mask, image_origin in tqdm(test_loader):
            image, mask = image.to(device), mask.long().to(device)

            # 추론
            output = model(image)

            # Softmax
            probs = F.softmax(output, dim=1).data.cpu().numpy()

            # loss 계산
            loss = criterion(output, mask)
            total_loss += loss

            # Dense CRF 적용
            if args['test_data_transform_dcrf']:
                with Pool(args['test_data_transform_dcrf_num_workers']) as pool:
                    probs = np.array(pool.map(DCRF_SubRoutine, zip(image_origin, probs, repeat(args)))) # pool에 dictionary를 전달하려면 repeat 필요

            # 추론 결과 segmentation 생성
            output = np.argmax(probs, axis=1)
            mask = mask.detach().cpu().numpy()

            # 추론 결과 누적
            hist = add_hist(hist, mask, output, n_class=n_class)

        # score 계산
        avrg_loss = total_loss / len(test_loader)
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes: round(IoU, 4)} for IoU, classes in zip(IoU, test_dataset.Categories())]
        print(f'Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}, IoU by class : {IoU_by_class} ')


if __name__ == '__main__':
    # config file 로드
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='Configs/UNetPP_Effib4_DiceCE_AdamW_ObMix_pseudo.json', type=str, help="Train.py config.json")
    with open(parser.parse_args().config, 'r') as f:
        args = json.load(f)

    # 하드코딩으로 config 값을 수정하고싶다면 아래와 같이 수정할 수 있습니다.
    '''
    args['path_dataset_root'] = '../input/data/'
    args['path_json_valid'] = '../input/data/val.json'

    args['random_fix'] = True
    args["random_seed"] = 21
    
    args['model'] = 'UNetPP_Efficientb4' # referenced from Modules/Models.py
    
    args['test_path_checkpoint'] = './Projects/UNetPP_Effib4_aug2/best_score.pt'
    args['test_path_submission'] = './Projects/UNetPP_Effib4_aug2/submission.csv'

    args['test_model_batch_size'] = 8
    args['test_data_num_workers'] = 2
    args['test_data_loading_mode'] = 'realtime'  # 'preload' or 'realtime'
    args['test_data_transform_preprocess'] = 'Default' # referenced from Modules/Transform_Train.py
    args['test_data_transform_tta'] = 'HorizontalFlip_Rotate90' # referenced from Modules/Transform_TTA.py
    args['test_data_target_size'] = 256 # predict size
    '''

    freeze_support()
    Main(args)


