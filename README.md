# Train
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

# Test
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

--------

# HRNet OCR 학습 필요 사항
- HRNet OCR 학습을 위해선 imagenet pretrained model을 다운받아야합니다.
- 이 [링크](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk)에서 hrnetv2_w48_imagenet_pretrained.pth 모델을 다운받아 `Modules/Hrnet_Sources/` 아래 두어야 합니다. 
