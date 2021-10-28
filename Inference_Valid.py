'''
이 코드는 다음 기능을 포함합니다.
 - pytorch 기반의 Semantic Segmentation Model 을 Inference 해서 Validation set으로 TTA 등의 score를 비교실험합니다.
 - config.json 파일을 통해 추론 인자를 제어할 수 있습니다.

 - 사용법
    Inference_Valid.py --config config_valid.json

 - 출력
    [작성중]

 - 종속 파일 명세
    - Modules/Data.py : Custom Dataset 클래스들이 정의되어 있습니다. (현재는 부스트캠프 컴피티션용 COCO format만 지원합니다.)
    - Modules/Transform_Preprocess.py : 이미지 전처리를 위한 Transform들이 정의되어 있습니다.
    - Modules/Models.py : Semantic Segmentation 모델들이 정의되어 있습니다.
    - Modules/Transform_TTA.py : TTA를 위한 Transform 들이 정의되어 있습니다.
    - Utils/Tools.py : 각종 편의기능들이 정의되어 있습니다.

작성자 JiSeong Kim
최초 작성일 2021-10-28
'''
