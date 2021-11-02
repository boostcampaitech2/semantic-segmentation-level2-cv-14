'''
이 코드는 다음 기능을 포함합니다.
 - pytorch 기반의 Semantic Segmentation Model 을 학습합니다.
 - config.json 파일을 통해 학습 인자를 제어할 수 있습니다.

 - 사용법
    Train.py --config Configs/UNetPP_Efficientb4_aug_train.json

 - 출력
    'path_project_root' 값으로 지정한 경로에 다음과 같은 파일이 생성됩니다.
        best_score.pt : 모델이 가장 높은 validation score를 기록했던 시점의 weight 파일입니다.
        logs/
            best_score.log : 모델이 가장 높은 validation score를 갱신했던 시점의 epoch와 점수를 기록한 파일입니다.
            config.json : 모델 학습에 사용됐던 config.json 파일 사본입니다.
            train.log : 학습 출력 기록입니다.
            valid.log : 검증 출력 기록입니다.

 - 종속 파일 명세
    - Modules/Data.py : Custom Dataset 클래스가 정의되어 있습니다. (현재는 부스트캠프 컴피티션용 COCO format만 지원합니다.)
    - Modules/Models.py : Semantic Segmentation 모델들이 정의되어 있습니다.
    - Modules/Losses.py : 학습에 사용될 Loss들이 정의되어있습니다.
    - Modules/Optimizer.py : 학습에 사용될 Optimizer들이 정의되어 있습니다.
    - Modules/Transform_Preprocess.py : 이미지 전처리를 위한 Transform들이 정의되어 있습니다.
    - Utils/Tools.py : 각종 편의기능들이 정의되어 있습니다.
    - Utils/utils.py : Semantic Segmentation score 계산을 위한 도구들이 정의되어 있습니다.

 - 기본 config_train.json 요소 명세
    - path_dataset_root : 데이터셋이 저장된 root를 정의합니다.
    - path_train_json : train 데이터 json 파일을 정의합니다.
    - path_valid_json : test 데이터 json 파일을 정의합니다.
    - path_project_root : 학습될 모델이 저장될 디렉토리를 정의합니다.
    - random_fix : 난수 고정 여부를 정의합니다.
    - random_seed : 고정할 seed를 정의합니다.
    - model : Semantic Segmentation 모델을 정의합니다.
    - model_num_epochs : 학습 epoch 수를 정의합니다.
    - model_batch_size : 모델의 배치 크기를 정의합니다.
    - loss : 학습에 사용할 loss를 정의합니다.
    - optimizer : 학습에 사용할 optimizer를 정의합니다.
    - optimizer_learning_rate : 학습에 사용할 learning rate를 정의합니다.
    - optimizer_weight_decay : 학습에 사용할 weight decay를 정의합니다.
    - data_num_workers : data loader 가 사용할 프로세스 수를 정의합니다.
    - data_loading_mode : 데이터를 미리 로드하거나, 실시간으로 로드하는 모드를 정의합니다. 'preload' 또는 'realtime' 으로 선택할 수 있습니다.
    - data_train_transform : 학습에 사용될 전처리 transform을 정의합니다.
    - data_valid_transform : 검증에 사용될 전처리 transform을 정의합니다. ('Default'를 추천합니다.)


작성자 JiSeong Kim
최초 작성일 2021-10-28
'''

# Build-in
from multiprocessing import freeze_support
import os
import json
import argparse

# External
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Custom
from Modules import Data, Models, Transforms_Preprocess, Losses, Optimizers
from Utils.utils import label_accuracy_score, add_hist
from Utils import Tools



def Main(args):
    # 프로젝트 디렉토리 생성
    project_path = args['path_project_root']
    project_log_path = os.path.join(args['path_project_root'], "logs")
    Tools.CreateDirectory(project_path)
    Tools.CreateDirectory(project_log_path)

    # 현재 args 기록
    with open(os.path.join(project_log_path, "config.json"), 'w') as f:
        f.write(json.dumps(args, indent=2))
    
    # 기존 log파일이 있다면 삭제
    if os.path.isfile(os.path.join(project_log_path, "train.log")):
        os.remove(os.path.join(project_log_path, "train.log"))
    if os.path.isfile(os.path.join(project_log_path, "valid.log")):
        os.remove(os.path.join(project_log_path, "valid.log"))
    if os.path.isfile(os.path.join(project_log_path, "best_score.log")):
        os.remove(os.path.join(project_log_path, "best_score.log"))

    # seed 고정
    if args['random_fix']:
        Tools.Fix(args['random_seed'])

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 모델 정의
    model = getattr(Models, args['model'])()

    # device 할당
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        args['train_model_batch_size'] = args['train_model_batch_size'] * torch.cuda.device_count()
    model = model.to(device)

    # Train, Valid 데이터셋 정의
    train_dataset = Data.DataSet_Trash(args['path_json_train'], args['path_dataset_root'], loading_mode=args['train_data_loading_mode'], transforms=getattr(Transforms_Preprocess, args['train_data_transform_preprocess_train'])())
    valid_dataset = Data.DataSet_Trash(args['path_json_valid'], args['path_dataset_root'], loading_mode=args['train_data_loading_mode'], transforms=getattr(Transforms_Preprocess, args['train_data_transform_preprocess_valid'])())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args['train_model_batch_size'], shuffle=True, num_workers=args['train_data_num_workers'], drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args['train_model_batch_size'], shuffle=False, num_workers=args['train_data_num_workers'])

    # Loss function 정의
    criterion = getattr(Losses, args['train_loss'])()

    # Optimizer 정의
    optimizer = getattr(Optimizers, args['train_optimizer'])(model=model, lr=args['train_optimizer_learning_rate'], weight_decay=args['train_optimizer_weight_decay'])

    # 학습
    n_class = len(train_dataset.Categories())
    best_mIoU = 0

    for epoch in range(1, args['train_model_num_epochs'] + 1):
        for stage in ['train', 'valid']:
            if stage == 'train':
                model.train()
                loader = train_loader
                color = 95
            elif stage == 'valid':
                model.eval()
                loader = valid_loader
                color = 96

            total_loss = 0
            hist = np.zeros((n_class, n_class))
            for images, masks, _ in tqdm(loader):
                images, masks = images.to(device), masks.long().to(device)

                # inference
                if stage == 'train':
                    outputs = model(images)
                elif stage == 'valid':
                    with torch.no_grad():
                        outputs = model(images)

                # loss 계산
                loss = criterion(outputs, masks)
                total_loss += loss

                # 학습
                if stage == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 추론 결과 segmentation 생성
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()

                # 추론 결과 누적
                hist = add_hist(hist, masks, outputs, n_class=n_class)

            # score 계산
            avrg_loss = total_loss / len(loader)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            IoU_by_class = [{classes: round(IoU, 4)} for IoU, classes in zip(IoU, train_dataset.Categories())]
            print(f'\n\033[{color}m' + f"[{epoch}/{args['train_model_num_epochs']}] {stage} # Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}, IoU by class : {IoU_by_class} " + '\033[0m')

            # log 저장
            with open(os.path.join(project_log_path, stage+".log"), 'a') as f:
                f.write(f"Epoch : {epoch}, loss : {round(avrg_loss.item(), 4)}, acc : {round(acc, 4)}, acc_cls : {acc_cls}, mIoU : {mIoU}, fwavacc : {fwavacc}, IoU_by_class : {IoU_by_class}\n")

            if stage == 'valid':
                # Best score 모델, 로그 저장
                if mIoU > best_mIoU:
                    print(f"Best performance at epoch: {epoch}, Save model in {project_path}")
                    best_mIoU = mIoU
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module, os.path.join(project_path, "best_score.pt"))
                    else:
                        torch.save(model, os.path.join(project_path, "best_score.pt"))
                    with open(os.path.join(project_log_path, "best_score.log"), 'a') as f:
                        f.write(f"Epoch : {epoch}, Best_mIoU : {best_mIoU}\n")

if __name__ == '__main__':
    # config file 로드
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./Configs/PAN_ResNext101.json', type=str, help="Train.py config.json")
    with open(parser.parse_args().config, 'r') as f:
        args = json.load(f)

    # 하드코딩으로 config 값을 수정하고싶다면 아래와 같이 수정할 수 있습니다.
    '''
    args['path_dataset_root'] = '../input/data/'
    args['path_json_train'] = '../input/data/train.json'
    args['path_json_valid'] = '../input/data/val.json'
    args['path_project_root'] = './Projects/UNetPP_Effib4_aug_temp' # 새로 생성될 디렉토리를 지정합니다.

    args['random_fix'] = True
    args['random_seed'] = 21

    args['model'] = 'UNetPP_Efficientb4' # referenced from Modules/Models.py
    args['train_model_num_epochs'] = 200
    args['train_model_batch_size'] = 10

    args['train_loss'] = 'CrossEntropyLoss' # referenced from Modules/Losses.py

    args['train_optimizer'] = 'Adam' # referenced from Modules/Optimizers.py
    args['train_optimizer_learning_rate'] = 0.0001
    args['train_optimizer_weight_decay'] = 1e-6

    args['train_data_num_workers'] = 2 # data_loading_mode가 preload 인 경우 작은 값(2정도)을 할당해야 OOM이 발생하지 않습니다.
    args['train_data_loading_mode'] = 'realtime' # 'preload' or 'realtime'
    args['train_data_transform_train'] = 'HorizontalFlip_Rotate90' # referenced from Modules/Transforms.py
    args['train_data_transform_valid'] = 'Default' # referenced from Modules/Transforms.py
    '''

    freeze_support()
    Main(args)