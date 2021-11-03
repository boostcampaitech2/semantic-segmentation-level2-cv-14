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
    - Modules/Transform_Preprocess.py : 이미지 전처리를 위한 Transform들이 정의되어 있습니다.
    - Modules/Models.py : Semantic Segmentation 모델들이 정의되어 있습니다.
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
from itertools import repeat
from multiprocessing import freeze_support, Pool
import argparse
import json
import re

# External
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as alb
import ttach as tta
from pycocotools import mask as coco_mask
from skimage import measure
import cv2

# Custom
from Modules import Data, Models, Transforms_Preprocess, Transforms_TTA
from Modules.Transforms_AfterProcess import DCRF_SubRoutine
from Utils import Tools


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

    # Train, Valid 데이터셋 정의
    test_dataset = Data.DataSet_Trash(args['path_json_test'], args['path_dataset_root'], loading_mode=args['test_data_loading_mode'], transforms=getattr(Transforms_Preprocess, args['test_data_transform_preprocess'])(), stage='test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args['test_model_batch_size'], shuffle=False, num_workers=args['test_data_num_workers'])

    transform = alb.Compose([alb.Resize(args['test_data_target_size'], args['test_data_target_size'])])


    model.eval()
    with torch.no_grad():
        image_names = []
        preds = np.empty((0, args['test_data_target_size'] ** 2), dtype=np.long)

        for image, image_name, image_origin, image_id in tqdm(test_loader):
            image = image.to(device)

            # 추론
            output = model(image)

            # Softmax
            probs = F.softmax(output, dim=1).data.cpu().numpy()

            # Dense CRF 적용
            if args['test_data_transform_dcrf']:
                with Pool(args['test_data_transform_dcrf_num_workers']) as pool:
                    probs = np.array(pool.map(DCRF_SubRoutine, zip(image_origin, probs, repeat(args))))  # pool에 dictionary를 전달하려면 repeat 필요

            # 추론 결과 segmentation 생성
            output = np.argmax(probs, axis=1)

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(image.cpu().numpy()), output):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], args['test_data_target_size'] ** 2]).astype(int)
            preds = np.vstack((preds, oms))

            image_names += image_name

            # 슈도 레이블링
            if args["test_pseudo_labeling"]:
                if 'annotations' not in locals():
                    annotations = []

                for batch, id in zip(probs, image_id):
                    mask = np.zeros_like(batch, dtype=np.uint8)
                    mask[np.where(batch == np.max(batch, axis=0))] = 255
                    mask = np.pad(mask, ((0, 0), (1, 1), (1, 1)), 'constant')
                    for idx, mask_cls in enumerate(mask):
                        if idx == 0: # 배경 무시
                            continue

                        fortran_ground_truth_binary_mask = np.asfortranarray(mask_cls)
                        encoded_ground_truth = coco_mask.encode(fortran_ground_truth_binary_mask)
                        ground_truth_area = int(coco_mask.area(encoded_ground_truth))
                        ground_truth_bounding_box = list(coco_mask.toBbox(encoded_ground_truth))
                        contours = measure.find_contours(mask_cls, 0.5)

                        segmentation = []
                        for object in contours:
                            object = np.flip(object, axis=1).astype(int)
                            points = object.ravel().tolist()
                            segmentation.append(points)

                        if not segmentation or ground_truth_area < args["test_pseudo_labeling_threshold_area"]:
                            continue

                        annotation = {"image_id":id.item(), "category_id":idx, "segmentation":segmentation, "area":ground_truth_area, "bbox":ground_truth_bounding_box, "iscrowd":0}
                        annotations.append(annotation)

    # submission.csv 생성
    submission = pd.DataFrame({'image_id': [], 'PredictionString': []})
    for file_name, string in zip(image_names, preds):
        submission = submission.append({"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())}, ignore_index=True)
    submission.to_csv(args['test_path_submission'], index=False)

    # 슈도 레이블링
    if args["test_pseudo_labeling"]:
        with open(args["path_json_test"], 'r') as f:
            js_test = json.load(f)

        with open(args["path_json_train"], 'r') as f:
            js_train = json.load(f)

        idx_image = len(js_train["images"])
        for idx, image in enumerate(js_test["images"]):
            image["id"] += idx_image
            js_train["images"].append(image)

        idx_ann = len(js_train["annotations"])
        for idx, ann in enumerate(annotations):
            ann["image_id"] += idx_image
            ann["id"] = idx + idx_ann
            js_train["annotations"].append(ann)

        with open(args["test_pseudo_labeling_output_path"], 'w') as f:
            jdata = json.dump(js_train,f, indent=1)


if __name__ == '__main__':
    # config file 로드
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='Configs/UNetPP_Effib4_DiceCE_AdamW.json', type=str, help="Train.py config.json")
    with open(parser.parse_args().config, 'r') as f:
        args = json.load(f)

    # 하드코딩으로 config 값을 수정하고싶다면 아래와 같이 수정할 수 있습니다.
    '''
    args['path_dataset_root'] = '../input/data/'
    args['path_json_test'] = '../input/data/test.json'

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


