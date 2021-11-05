# 재활용 쓰레기 Semantic Segmentation

<img src="https://user-images.githubusercontent.com/44287798/140461430-78e5cd84-2162-4f98-9d27-bbc3a8580f90.png" width="400">  <img src="https://user-images.githubusercontent.com/44287798/140461384-a0a91b44-da3a-4b81-95cb-ec508b978aa7.png" width="400"> 


## 프로젝트 개요

우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있고 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있다.
 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나로, 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문이다.
 
우리는 사진에서 쓰레기를 탐지하는 모델을 만들어 이러한 문제점을 해결해보고자 한다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋을 사용한다. 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것이다.

## 팀원 소개
팀명: Machine==우리조 

||이름|역할|github|
|--|------|---|---|
|😙|김범수|PSPNet, HRNet_fcn, SwinT 실험|https://github.com/HYU-kbs|
|🤗|김준태|FCN, FPN, DeepLab V3+ 실험|https://github.com/sronger|
|😎|김지성|공용 실험 플랫폼 개발, UNet++ 실험, Pseudo Labeling 도구 개발, Object Mix Augmentation 도구 개발|https://github.com/intelli8786|
|😆|백종원|Unet++, DeepLab V3+ 실험|https://github.com/Baek-jongwon|
|😊|정소희|PSPNet, DeepLab V3+, SEResNext101+CBAM 실험|https://github.com/SoheeJeong|
|😄|홍지연|Unet, Hrnet OCR 실험|https://github.com/hongjourney|


## 모델 성능 및 config file

학습된 모델에 대한 설명과 성능, 각 모델에 대한 config file의 위치를 표로 나타내었다.
config file은 json 형식으로, hyperparameter, model architecture, optimizer, scheduler, train/test dataset 등 모델에 대한 전반적인 학습 정보를 포함한다. 

|모델|config|
|------|---|
|DeepLabV3P_Effib4_NoAug|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/DeepLabV3P_Effib4_NoAug.json)|
|Hrnet_Ocr_NoAug|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/Hrnet_Ocr_NoAug.json)|
|Hrnet_Ocr_aug|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/Hrnet_Ocr_aug.json)|
|Hrnet_Ocr_aug_gridmask|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/Hrnet_Ocr_aug_gridmask.json)|
|Hrnet_Ocr_aug_pseudo|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/Hrnet_Ocr_aug_pseudo.json)|
|Hrnet_Ocr_aug_gridmask_pseudo|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/Hrnet_Ocr_aug_gridmask_pseudo.json)|
|Hrnet_Ocr_aug_gridmask_CLAHE_pseudo|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/Hrnet_Ocr_aug_gridmask_CLAHE_pseudo.json)|
|Hrnet_Ocr_aug_gridmask_CLAHE_pseudo_objectMix|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/Hrnet_Ocr_aug_gridmask_CLAHE_pseudo_objectMix.json)|
|PAN_ResNext101|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/PAN_ResNext101.json)|
|PAN_ResNext101_pseudo|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/PAN_ResNext101_pseudo.json)|
|UNetPP_Effib3_DiceCE_AdamW|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib3_DiceCE_AdamW.json)|
|UNetPP_Effib3_DiceCE_AdamW_pseudo|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib3_DiceCE_AdamW_pseudo.json)|
|UNetPP_Effib4|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib4.json)|
|UNetPP_Effib4_AdamW|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib4_AdamW.json)|
|UNetPP_Effib4_DiceCE|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib4_DiceCE.json)|
|UNetPP_Effib4_DiceCE_AdamW|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib4_DiceCE_AdamW.json)|
|UNetPP_Effib4_DiceCE_AdamW_ObMix|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib4_DiceCE_AdamW_ObMix.json)|
|UNetPP_Effib4_DiceCE_AdamW_ObMix_pseudo|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib4_DiceCE_AdamW_ObMix_pseudo.json)|
|UNetPP_Effib4_DiceCE_AdamW_pseudo|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib4_DiceCE_AdamW_pseudo.json)|
|UNetPP_Effib4_aug|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib4_aug.json)|
|UNetPP_Effib4_aug_AdamW|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib4_aug_AdamW.json)|
|UNetPP_Effib4_aug_DiceCE|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib4_aug_DiceCE.json)|
|UNetPP_Effib4_aug_DiceCE_AdamW|[config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/blob/master/Configs/UNetPP_Effib4_aug_DiceCE_AdamW.json)|

## Code Description

##### Train.py
 - Semantic Segmentation Model 을 학습합니다.
 - config.json 파일을 통해 학습 인자를 제어할 수 있습니다.

 - 사용법
   - python Train.py --config Configs/UNetPP_Effib4_DiceCE_AdamW_ObMix.json

 - 출력
   - 'path_project_root' 값으로 지정한 경로에 다음과 같은 파일이 생성됩니다.
      - best_score.pt : 모델이 가장 높은 validation score를 기록했던 시점의 weight 파일입니다.
      - logs/
         - best_score.log : 모델이 가장 높은 validation score를 갱신했던 시점의 epoch와 점수를 기록한 파일입니다.
         - config.json : 모델 학습에 사용됐던 config.json 파일 사본입니다.
         - train.log : 학습 출력 기록입니다.
         - valid.log : 검증 출력 기록입니다.

##### Inference_Test.py
 - Semantic Segmentation Model 을 Inference 해서 submission.csv 파일을 생성합니다.
 - config.json 파일을 통해 추론 인자를 제어할 수 있습니다.
    - TTA와 Dense CRF등의 후처리를 적용할 수 있습니다.

 - 사용법
   - python Inference_Test.py --config Configs/UNetPP_Effib4_DiceCE_AdamW_ObMix.json

 - 출력
   - 'path_save' 값으로 지정한 경로에 다음과 같은 파일이 생성됩니다.
      - submission.csv : 부스트캠프 컴피티션 형식에 맞추어 생성된 Segmentation 결과 파일입니다.

##### Inference_Valid.py
 - pSemantic Segmentation Model 을 Inference 해서 Validation 데이터를 통해 모델의 성능을 확인합니다.
 - config.json 파일을 통해 추론 인자를 제어할 수 있습니다.
   - TTA와 Dense CRF등의 후처리를 적용할 수 있습니다.

 - 사용법
   - python Inference_Valid.py --config Configs/UNetPP_Effib4_DiceCE_AdamW_ObMix.json

 - 출력
   - Validation 데이터로 추론했을때의 모델의 성능을 출력합니다.

<details open>
 <summary> 파일 명세 </summary>
 
    - Train.py : 학습을 위한 기능이 정의되어 있습니다. 이 파일을 사용해 Semantic Segmentation 모델을 학습할 수 있습니다.

    - Inference_Valid.py : Validation 데이터를 사용해 TTA, Dence CRF등을 실험하기 위한 기능이 정의되어 있습니다. 이 파일을 사용해 Validation 데이터에서 후처리 실험을 할 수 있습니다.
    
    - Inference_Test.py : Test 데이터를 사용해 Submission 파일을 생성하는 기능이 정의되어 있습니다. 이 파일을 사용해 TTA, Dence CRF 등의 후처리를 적용하고, 그 결과로 Submission 파일을 생성할 수 있습니다.

    - Modules/Data.py : Custom Dataset 클래스가 정의되어 있습니다. (현재는 부스트캠프 컴피티션용 COCO format만 지원합니다.)

    - Modules/Models.py : Semantic Segmentation 모델들이 정의되어 있습니다.

    - Modules/Losses.py : 학습에 사용될 Loss들이 정의되어있습니다.

    - Modules/Optimizer.py : 학습에 사용될 Optimizer들이 정의되어 있습니다.

    - Modules/Transform_Preprocess.py : 이미지 전처리를 위한 Transform들이 정의되어 있습니다.

    - Modules/Transform_TTA.py : TTA를 위한 Transform 들이 정의되어 있습니다.

    - Modules/Transform_AfterProcess.py : TTA를 위한 Transform 들이 정의되어 있습니다.

    - Utils/Tools.py : 각종 편의기능들이 정의되어 있습니다.

    - Utils/utils.py : Semantic Segmentation score 계산을 위한 도구들이 정의되어 있습니다.
</details>
<details open>
 <summary> config 요소 명세 </summary>
 
    - "path_dataset_root" : 데이터셋이 저장된 root를 정의합니다.
    - "path_json_train" : train 데이터 json 파일을 정의합니다.
    - "path_json_valid" : valid 데이터 json 파일을 정의합니다.
    - "path_json_test" : test 데이터 json 파일을 정의합니다.
    - "path_project_root" : 학습될 모델이 저장될 디렉토리를 정의합니다.

    - "random_fix": true, : 난수 고정 여부를 정의합니다.
    - "random_seed": 21, : 고정할 seed를 정의합니다.

    - "model" : Semantic Segmentation 모델을 정의합니다.

    - "train_model_pretrained" : 사전학습된 모델을 사용할것인지 여부를 정의합니다.
    - "train_model_pretrained_path" : 사전학습된 모델의 경로를 정의합니다.
    - "train_model_num_epochs" : 학습 epoch 수를 정의합니다.
    - "train_model_batch_size" : 학습시의 모델의 배치 크기를 정의합니다.
    - "train_loss": "DiceCELoss", : 학습에 사용할 loss를 정의합니다.
    - "train_optimizer" : 학습에 사용할 optimizer를 정의합니다.
    - "train_optimizer_learning_rate" : 학습에 사용할 learning rate를 정의합니다.
    - "train_optimizer_weight_decay" : 학습에 사용할 weight decay를 정의합니다.
    - "train_data_num_workers" : data loader 가 사용할 프로세스 수를 정의합니다.
    - "train_data_loading_mode" : 데이터를 미리 로드하거나, 실시간으로 로드하는 모드를 정의합니다. 'preload' 또는 'realtime' 으로 선택할 수 있습니다.
    - "train_data_transform_preprocess_train" : 학습에 사용될 전처리 transform을 정의합니다.
    - "train_data_transform_preprocess_train_object_aug" : Object Mix Augmentation 적용 여부를 정의합니다.
    - "train_data_transform_preprocess_valid" : 검증에 사용될 전처리 transform을 정의합니다. ('Default'를 추천합니다.)

    - "test_path_checkpoint": "./Projects/UNetPP_Effib4_DiceCE_AdamW_ObMix_pseudo/best_score.pt",
    - "test_path_submission" : "./Projects/UNetPP_Effib4_DiceCE_AdamW_ObMix_pseudo/submission.csv",
    - "test_model_batch_size": 10, : 테스트시의 모델의 배치 크기를 정의합니다.
    - "test_data_num_workers": 4, : data loader 가 사용할 프로세스 수를 정의합니다.
    - "test_data_loading_mode" : 데이터를 미리 로드하거나, 실시간으로 로드하는 모드를 정의합니다. 'preload' 또는 'realtime' 으로 선택할 수 있습니다.
    - "test_data_target_size" : submission파일에 적용할 출력크기를 정의합니다.
    - "test_data_transform_preprocess" : Test에 사용할 이미지변환을 정의합니다. ('Default'를 추천합니다.)
    - "test_data_transform_tta" : 적용할 Test Time Augmentation을 정의합니다.
    - "test_data_transform_dcrf" : Dense CRF의 적용 여부를 정의합니다.
    - "test_data_transform_dcrf_num_workers" : Dense CRF 연산을 위한 sub process 개수를 정의합니다.
    - "test_data_transform_dcrf_iter" : 10, : Dense CRF 연산의 반복회수를 정의합니다.
    - "test_data_transform_dcrf_gau_sxy" : Dense CRF 연산의 Pairwise Gaussian sxy 값을 정의합니다.
    - "test_data_transform_dcrf_gau_compat" : Dense CRF 연산의 Pairwise Gaussian compat 값을 정의합니다.
    - "test_data_transform_dcrf_bi_sxy" : Dense CRF 연산의 Bilateral sxy 값을 정의합니다.
    - "test_data_transform_dcrf_bi_srgb" : Dense CRF 연산의 Bilateral srgb 값을 정의합니다.
    - "test_data_transform_dcrf_bi_compat" : Dense CRF 연산의 Bilateral compat 값을 정의합니다.

    - "test_pseudo_labeling" : pseudo labeling 적용 유무를 정의합니다.
    - "test_pseudo_labeling_threshold_area" : pseudo labeling 이후 추론된 객체가 이 threshold 이하일 경우 제외합니다.
    - "test_pseudo_labeling_output_path" : pseudo labeling 결과로 저장될 coco format의 json파일 경로를 정의합니다.
</details>


### HRNet OCR 학습 필요 사항
- HRNet OCR 학습을 위해선 imagenet pretrained model을 다운받아야합니다.
- 이 [링크](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk)에서 hrnetv2_w48_imagenet_pretrained.pth 모델을 다운받아 `Modules/Hrnet_Sources/` 아래 두어야 합니다. 



----

### 실험 결과 도식화

다음 [링크](https://www.edrawmind.com/online/map.html?sharecode=6184ef514b5544a49698555)는 실험 목록을 마인드맵 형식으로 도식화한 것입니다.
![Segmentation31-web-16349014493121112222112](https://user-images.githubusercontent.com/44287798/140483293-89c769c7-37fb-4b21-85b3-d14e3e882c57.png)


